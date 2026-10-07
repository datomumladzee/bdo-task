"""Checks the HR assistant runs before creating a leave request (Policy Article 12.3).

validate_request() first runs blocking checks (employee, leave type, dates,
zero days) and stops at the first failure, because later checks need valid
input. It then runs every policy check and collects all violations, so the
employee hears about every problem at once.

Actions:
    reject   - the request cannot be made as asked; change it (see `alternative`).
    redirect - the request belongs to another channel (HR, HR portal, manager).

Violation codes:
    UNKNOWN_EMPLOYEE, UNKNOWN_LEAVE_TYPE, LEAVE_TYPE_NOT_SUPPORTED,
    DATES_INVALID, DATES_CROSS_YEAR, DATES_NEXT_YEAR, DATES_PAST_YEAR, ZERO_DAYS,
    PROBATION, NOTICE_PERIOD, SICK_LATE, SICK_FUTURE_UNCONFIRMED, MAX_CONTINUOUS,
    RESTRICTED_PERIOD, NO_ENTITLEMENT, INSUFFICIENT_BALANCE, SICK_PAID_LIMIT,
    REASON_REQUIRED, OVERLAP
"""

import sqlite3
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date, timedelta

from server.balance import get_balance
from server.calendar import (
    count_calendar_days,
    count_working_days,
    is_working_day,
    load_holidays,
    working_days_between,
)

REJECT = "reject"
REDIRECT = "redirect"

MAX_CONTINUOUS_ANNUAL_DAYS = 15  # Article 4.5
SICK_LATE_LIMIT_DAYS = 2  # Article 6.2
UNPAID_NOTICE_DAYS = 10  # Article 7.2
RESTRICTED_DEPARTMENT = "AUD"  # Article 4.6

STATUS_KA = {"pending": "განხილვის პროცესში", "approved": "დამტკიცებული"}

# Leave types the assistant explains but never creates (Articles 8, 9, 10).
_REDIRECT_TYPES: dict[str, tuple[str, str, str]] = {
    "BEREAVEMENT": (
        "8.3",
        (
            "გლოვის შვებულების მოთხოვნას ასისტენტი ვერ შექმნის. მოთხოვნა წარადგინეთ "
            "HR პორტალით ან ადამიანური რესურსების სამსახურის მეშვეობით, შვებულების "
            "პირველი დღიდან არაუგვიანეს 2 სამუშაო დღისა."
        ),
        "მოთხოვნაში მიუთითეთ ნათესაური კავშირი და გარდაცვალების თარიღი.",
    ),
    "STUDY": (
        "9.3",
        (
            "სასწავლო შვებულების მოთხოვნას ასისტენტი ვერ შექმნის. მოთხოვნა წარადგინეთ "
            "HR პორტალით ან HR-ის მეშვეობით, დაწყებამდე სულ მცირე 10 სამუშაო დღით ადრე."
        ),
        (
            "მიუთითეთ გამოცდის დასახელება და თარიღი. დამტკიცებულ გეგმასთან "
            "შესაბამისობას HR ამოწმებს."
        ),
    ),
    "PARENTAL": (
        "10.2",
        "მშობლის შვებულება თვითმომსახურების არხებით, მათ შორის ასისტენტით, არ წარდგება.",
        (
            "პირდაპირ მიმართეთ ადამიანური რესურსების სამსახურს, შვებულების სავარაუდო "
            "დაწყებამდე არაუგვიანეს 8 კვირით ადრე."
        ),
    ),
}


@dataclass(frozen=True)
class RuleViolation:
    code: str
    action: str
    message_ka: str
    article: str
    alternative: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    violations: tuple[RuleViolation, ...]
    days: int | None = None

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def codes(self) -> list[str]:
        return [v.code for v in self.violations]


def _fail(violation: RuleViolation) -> ValidationResult:
    return ValidationResult(violations=(violation,))


def validate_request(
    conn: sqlite3.Connection,
    *,
    employee_id: str,
    leave_type: str,
    start: date,
    end: date,
    today: date,
    comment: str | None = None,
    known_in_advance: bool = False,
) -> ValidationResult:
    """Check whether the assistant may create this request for the employee."""
    employee = conn.execute(
        "SELECT department_code, probation_end_date FROM employees WHERE employee_id = ?",
        (employee_id,),
    ).fetchone()
    if employee is None:
        return _fail(
            RuleViolation(
                "UNKNOWN_EMPLOYEE",
                REJECT,
                f"თანამშრომელი {employee_id} ვერ მოიძებნა.",
                "12.2",
            )
        )

    type_row = conn.execute(
        "SELECT day_unit, assistant_supported, policy_reference FROM leave_types WHERE code = ?",
        (leave_type,),
    ).fetchone()
    if type_row is None:
        return _fail(
            RuleViolation(
                "UNKNOWN_LEAVE_TYPE",
                REJECT,
                f"შვებულების სახე „{leave_type}“ არ არსებობს.",
                "3.1",
                "შვებულების სახეებია: ANNUAL, SICK, UNPAID, BEREAVEMENT, STUDY, PARENTAL.",
            )
        )
    if not type_row["assistant_supported"]:
        article, message, alternative = _REDIRECT_TYPES.get(
            leave_type,
            (
                type_row["policy_reference"] or "12.3",
                "ამ სახის მოთხოვნას ასისტენტი ვერ შექმნის.",
                "მიმართეთ HR პორტალს ან ადამიანური რესურსების სამსახურს.",
            ),
        )
        return _fail(
            RuleViolation("LEAVE_TYPE_NOT_SUPPORTED", REDIRECT, message, article, alternative)
        )

    if date_violation := _check_dates(start, end, today):
        return _fail(date_violation)

    holidays = load_holidays(conn)
    if type_row["day_unit"] == "calendar":
        days = count_calendar_days(start, end)
    else:
        days = count_working_days(start, end, holidays)
    if days == 0:
        return _fail(
            RuleViolation(
                "ZERO_DAYS",
                REJECT,
                "არჩეულ პერიოდში სამუშაო დღე არ არის: მხოლოდ შაბათ-კვირა ან უქმე დღეებია.",
                "2.2",
                "აირჩიეთ პერიოდი, რომელიც სამუშაო დღეს მოიცავს.",
            )
        )

    # Policy checks: run all of them and collect every violation, so the
    # employee sees all problems at once (Article 12.3). The blocking checks
    # above stop early because these checks need valid input.
    checks = [
        _check_probation(leave_type, start, employee["probation_end_date"]),
        _check_notice(leave_type, start, days, today, holidays),
        _check_sick_timing(leave_type, start, today, known_in_advance, holidays),
        _check_max_continuous(conn, employee_id, leave_type, start, end, days, holidays),
        _check_restricted_period(leave_type, start, end, employee["department_code"]),
        _check_balance(conn, employee_id, leave_type, start.year, days),
        _check_reason(leave_type, comment),
        _check_overlap(conn, employee_id, start, end),
    ]
    # Keep only the checks that returned a violation (None means the rule passed).
    return ValidationResult(violations=tuple(v for v in checks if v), days=days)


def _check_dates(start: date, end: date, today: date) -> RuleViolation | None:
    """Articles 2.1, 12.2 and 12.3: one leave year, the current one."""
    if start > end:
        return RuleViolation(
            "DATES_INVALID",
            REJECT,
            "დაწყების თარიღი დასრულების თარიღზე გვიან ვერ იქნება.",
            "2.1",
        )
    if start.year != end.year:
        return RuleViolation(
            "DATES_CROSS_YEAR",
            REJECT,
            "შვებულება ერთი წლიდან მეორეში გადადის. თითოეული წლისთვის ცალკე მოთხოვნაა საჭირო.",
            "2.1",
            f"მოითხოვეთ შვებულება {start.year} წლის 31 დეკემბრის ჩათვლით, ხოლო მომდევნო "
            "წლის ნაწილი ცალკე მოთხოვნით წარადგინეთ HR პორტალით.",
        )
    if start.year > today.year:
        return RuleViolation(
            "DATES_NEXT_YEAR",
            REJECT,
            "ასისტენტი მომდევნო წლის თარიღებზე მოთხოვნას ვერ ქმნის.",
            "12.3",
            f"ასეთი მოთხოვნა HR პორტალით შეგიძლიათ წარადგინოთ {today.year} წლის 1 დეკემბრიდან.",
        )
    if start.year < today.year:
        return RuleViolation(
            "DATES_PAST_YEAR",
            REJECT,
            "ასისტენტი მოთხოვნას მხოლოდ მიმდინარე წლის თარიღებზე ქმნის.",
            "12.2",
            "წინა წლის საკითხებთან დაკავშირებით მიმართეთ ადამიანური რესურსების სამსახურს.",
        )
    return None


def _check_probation(leave_type: str, start: date, probation_end: str) -> RuleViolation | None:
    """Article 4.3: no ANNUAL leave on any day up to the last day of probation."""
    if leave_type != "ANNUAL":
        return None
    last_day = date.fromisoformat(probation_end)
    if start > last_day:
        return None
    return RuleViolation(
        "PROBATION",
        REJECT,
        f"გამოსაცდელი ვადის განმავლობაში (ბოლო დღე: {last_day.isoformat()}) "
        "ყოველწლიური შვებულების გამოყენება არ შეიძლება.",
        "4.3",
        f"ყოველწლიური შვებულება შეიძლება დაიწყოს {(last_day + timedelta(days=1)).isoformat()}-დან. "
        "გამონაკლის შემთხვევაში, მაგალითად ოჯახური მიზეზით, პირდაპირ მიმართეთ ადამიანური "
        "რესურსების სამსახურს. ავადმყოფობის და უხელფასო შვებულებაზე ეს შეზღუდვა არ ვრცელდება.",
    )


def _required_notice(leave_type: str, days: int) -> tuple[int, str] | None:
    """Minimum notice in working days and its article, or None if not applicable."""
    if leave_type == "UNPAID":
        return UNPAID_NOTICE_DAYS, "7.2"
    if leave_type == "ANNUAL":
        if days <= 5:
            return 5, "4.4"
        if days <= MAX_CONTINUOUS_ANNUAL_DAYS:
            return 15, "4.4"
        # Over 15 days the notice table defers to Article 4.5 (MAX_CONTINUOUS).
    return None


def _earliest_start(today: date, required: int, holidays: Collection[date], working: bool) -> date:
    day = today + timedelta(days=1)
    while working_days_between(today, day, holidays) < required or (
        working and not is_working_day(day, holidays)
    ):
        day += timedelta(days=1)
    return day


def _check_notice(
    leave_type: str, start: date, days: int, today: date, holidays: Collection[date]
) -> RuleViolation | None:
    """Articles 4.4 and 7.2: full working days between today and the first day."""
    notice = _required_notice(leave_type, days)
    if notice is None:
        return None
    required, article = notice
    actual = working_days_between(today, start, holidays)
    if actual >= required:
        return None
    earliest = _earliest_start(today, required, holidays, working=leave_type == "ANNUAL")
    alternative = f"ყველაზე ადრეული შესაძლო დაწყების თარიღია {earliest.isoformat()}."
    if leave_type == "ANNUAL":
        alternative += (
            " გადაუდებელი პირადი მიზეზის შემთხვევაში მოთხოვნა პირდაპირ უშუალო "
            "ხელმძღვანელს გაუგზავნეთ."
        )
    if start <= today:
        situation = "შვებულების დაწყების თარიღი უკვე დადგა ან გასულია."
    else:
        situation = f"ახლა მოთხოვნასა და დაწყებას შორის {actual} სამუშაო დღეა."
    return RuleViolation(
        "NOTICE_PERIOD",
        REJECT,
        f"{days} დღის შვებულებისთვის საჭიროა სულ მცირე {required} სამუშაო დღით ადრე "
        f"წარდგენა. {situation}",
        article,
        alternative,
    )


def _check_sick_timing(
    leave_type: str,
    start: date,
    today: date,
    known_in_advance: bool,
    holidays: Collection[date],
) -> RuleViolation | None:
    """Article 6.2: record within 2 working days after the first day; future
    sick leave only when the period is already known."""
    if leave_type != "SICK":
        return None
    if start < today:
        # The first day itself is not counted; today is.
        elapsed = count_working_days(start + timedelta(days=1), today, holidays)
        if elapsed > SICK_LATE_LIMIT_DAYS:
            return RuleViolation(
                "SICK_LATE",
                REDIRECT,
                "ავადმყოფობის შვებულება უნდა აღირიცხოს პირველი დღიდან არაუგვიანეს "
                "2 სამუშაო დღისა. ეს ვადა გასულია, ამიტომ ასისტენტი მოთხოვნას ვერ შექმნის.",
                "6.2",
                "მიმართეთ ადამიანური რესურსების სამსახურს.",
            )
    elif start > today and not known_in_advance:
        return RuleViolation(
            "SICK_FUTURE_UNCONFIRMED",
            REJECT,
            "მომავალი თარიღით ავადმყოფობის შვებულება აღირიცხება მხოლოდ მაშინ, როცა "
            "გაცდენის პერიოდი უკვე ცნობილია, მაგალითად ექიმის დოკუმენტით ან დაგეგმილი "
            "ოპერაციის შემთხვევაში.",
            "6.2",
            "დაადასტურეთ, რომ პერიოდი წინასწარ არის ცნობილი. სამედიცინო დეტალების "
            "მოწოდება საჭირო არ არის.",
        )
    return None


def _check_max_continuous(
    conn: sqlite3.Connection,
    employee_id: str,
    leave_type: str,
    start: date,
    end: date,
    days: int,
    holidays: Collection[date],
) -> RuleViolation | None:
    """Article 4.5: at most 15 working days of continuous ANNUAL leave.

    Pending or approved ANNUAL requests separated from this one only by
    weekends or holidays join it into one continuous leave, in both directions.
    """
    if leave_type != "ANNUAL":
        return None
    others = [
        (date.fromisoformat(r["start_date"]), date.fromisoformat(r["end_date"]), r["days"])
        for r in conn.execute(
            """
            SELECT start_date, end_date, days FROM leave_requests
            WHERE employee_id = ? AND leave_type = 'ANNUAL'
              AND status IN ('pending', 'approved')
            """,
            (employee_id,),
        )
    ]
    chain_start, chain_end, total = start, end, days
    extended = True
    while extended:
        extended = False
        for other in list(others):
            other_start, other_end, other_days = other
            before = other_end < chain_start and (
                working_days_between(other_end, chain_start, holidays) == 0
            )
            after = other_start > chain_end and (
                working_days_between(chain_end, other_start, holidays) == 0
            )
            if before or after:
                chain_start = min(chain_start, other_start)
                chain_end = max(chain_end, other_end)
                total += other_days
                others.remove(other)
                extended = True

    if total <= MAX_CONTINUOUS_ANNUAL_DAYS:
        return None
    if total == days:
        detail = f"ეს მოთხოვნა {days} სამუშაო დღეა."
    else:
        detail = (
            f"მიმდებარე მოთხოვნებთან ერთად უწყვეტი შვებულება {total} სამუშაო დღე გამოდის "
            f"({chain_start.isoformat()} – {chain_end.isoformat()})."
        )
    return RuleViolation(
        "MAX_CONTINUOUS",
        REDIRECT,
        "ერთი უწყვეტი ყოველწლიური შვებულება 15 სამუშაო დღეს ვერ აღემატება. " + detail,
        "4.5",
        "უფრო ხანგრძლივი შვებულებისთვის საჭიროა სტრუქტურული ერთეულის პარტნიორის ან "
        "დირექტორის წერილობითი თანხმობა; მოთხოვნა უშუალო ხელმძღვანელის მეშვეობით "
        "წარადგინეთ. სხვა შემთხვევაში შეამცირეთ დღეების რაოდენობა.",
    )


def _restricted_periods(year: int) -> list[tuple[date, date]]:
    """Article 4.6, both ends inclusive."""
    return [(date(year, 1, 15), date(year, 3, 15)), (date(year, 12, 1), date(year, 12, 20))]


def _check_restricted_period(
    leave_type: str, start: date, end: date, department_code: str
) -> RuleViolation | None:
    """Article 4.6: AUD employees' ANNUAL leave touching a restricted period."""
    if leave_type != "ANNUAL" or department_code != RESTRICTED_DEPARTMENT:
        return None
    for period_start, period_end in _restricted_periods(start.year):
        if start <= period_end and end >= period_start:
            return RuleViolation(
                "RESTRICTED_PERIOD",
                REDIRECT,
                f"აუდიტის დეპარტამენტისთვის {period_start.isoformat()} – "
                f"{period_end.isoformat()} შეზღუდული პერიოდია. თუ შვებულების რომელიმე დღე "
                "ამ პერიოდშია, მოთხოვნა HR პორტალით ან ასისტენტით არ იგზავნება.",
                "4.6",
                "მიმართეთ უშუალო ხელმძღვანელს, რომელიც საკითხს პროექტის პარტნიორთან "
                "შეათანხმებს, ან აირჩიეთ შეზღუდული პერიოდის გარეთ არსებული თარიღები.",
            )
    return None


def _check_balance(
    conn: sqlite3.Connection, employee_id: str, leave_type: str, year: int, days: int
) -> RuleViolation | None:
    """Articles 5.1, 6.4 and 7.1."""
    try:
        balance = get_balance(conn, employee_id, year, leave_type)
    except LookupError:
        return RuleViolation(
            "NO_ENTITLEMENT",
            REDIRECT,
            f"{year} წლისთვის {leave_type} ბალანსი ვერ მოიძებნა.",
            "5.3",
            "მიმართეთ ადამიანური რესურსების სამსახურს HR პორტალის „მოთხოვნა HR-ს“ ფორმით.",
        )
    available = balance.available_days
    if days <= available:
        return None
    if leave_type == "SICK":
        # Article 6.4: never say that sickness can no longer be recorded.
        return RuleViolation(
            "SICK_PAID_LIMIT",
            REDIRECT,
            f"მოთხოვნა ({days} დღე) აღემატება დარჩენილ ანაზღაურებად ავადმყოფობის დღეებს "
            f"({available}). ასეთ მოთხოვნას ასისტენტი ვერ შექმნის.",
            "6.4",
            "მიმართეთ ადამიანური რესურსების სამსახურს: ავადმყოფობის აღრიცხვა ამ შემთხვევაშიც "
            "შესაძლებელია, ხოლო აღრიცხვისა და ანაზღაურების პირობებს HR ინდივიდუალურად "
            "განსაზღვრავს.",
        )
    if leave_type == "UNPAID":
        article = "7.1"
        alternative = (
            "შეამცირეთ დღეების რაოდენობა. წელიწადში 30 დღეზე მეტი უხელფასო შვებულებისთვის "
            "საჭიროა მმართველი პარტნიორის თანხმობა; მიმართეთ ადამიანური რესურსების სამსახურს."
        )
    else:
        article = "5.1"
        alternative = "შეამცირეთ დღეების რაოდენობა ან განიხილეთ უხელფასო შვებულება."
    return RuleViolation(
        "INSUFFICIENT_BALANCE",
        REJECT,
        f"მოთხოვნილია {days} დღე, ხელმისაწვდომი ბალანსი კი {available} დღეა.",
        article,
        alternative,
    )


def _check_reason(leave_type: str, comment: str | None) -> RuleViolation | None:
    """Article 7.2: UNPAID requests need a short reason."""
    if leave_type != "UNPAID" or (comment and comment.strip()):
        return None
    return RuleViolation(
        "REASON_REQUIRED",
        REJECT,
        "უხელფასო შვებულების მოთხოვნაში მოკლედ უნდა მიუთითოთ მიზეზი.",
        "7.2",
        "მიუთითეთ მოკლე მიზეზი. ჯანმრთელობის დეტალების მოწოდება საჭირო არ არის.",
    )


def _check_overlap(
    conn: sqlite3.Connection, employee_id: str, start: date, end: date
) -> RuleViolation | None:
    """Article 12.3: no overlap with the employee's pending or approved requests."""
    clashes = conn.execute(
        """
        SELECT request_id, leave_type, start_date, end_date, status FROM leave_requests
        WHERE employee_id = ? AND status IN ('pending', 'approved')
          AND start_date <= ? AND end_date >= ?
        ORDER BY start_date
        """,
        (employee_id, end.isoformat(), start.isoformat()),
    ).fetchall()
    if not clashes:
        return None
    listed = "; ".join(
        f"№{r['request_id']} {r['leave_type']} {r['start_date']} – {r['end_date']} "
        f"({STATUS_KA[r['status']]})"
        for r in clashes
    )
    return RuleViolation(
        "OVERLAP",
        REJECT,
        f"მოთხოვნილი თარიღები ემთხვევა თქვენს სხვა მოთხოვნას: {listed}.",
        "12.3",
        "აირჩიეთ სხვა თარიღები. არსებული მოთხოვნის შესაცვლელად ან გასაუქმებლად "
        "გამოიყენეთ HR პორტალი.",
    )
