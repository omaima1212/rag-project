"""Generate a small, clean test PDF for the RAG pipeline.

Why a FICTIONAL company: if the document describes something real, the model
can answer from its training data and you cannot tell whether retrieval worked.
Every fact below is invented, so a correct answer proves the fact came from the
document rather than from the model's memory.

    python tools\make_test_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer)

OUT = Path(__file__).parent.parent / "data" / "northvale_handbook.pdf"

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=11,
                      leading=16, spaceAfter=10)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=17, spaceAfter=12)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                    spaceBefore=14, spaceAfter=8)

S = [
    ("h1", "Northvale Analytics — Company Handbook 2026"),

    ("h2", "1. About Northvale"),
    ("p", "Northvale Analytics was founded in March 2019 by Ingrid Halvorsen "
          "and Tomas Reyes in Bergen, Norway. The company builds forecasting "
          "software for regional electricity grids. As of January 2026 it "
          "employs 148 people across three offices: Bergen, Lisbon and "
          "Montreal. The Bergen office is the headquarters and holds the "
          "engineering and research teams."),
    ("p", "The company's flagship product is called Tidewater. It predicts "
          "electricity demand up to fourteen days ahead using weather data, "
          "historical consumption and industrial scheduling information. "
          "Tidewater is used by eleven grid operators across northern Europe."),

    ("h2", "2. Working hours and leave"),
    ("p", "The standard working week is 37 hours. Core hours, during which all "
          "staff are expected to be reachable, run from 10:00 to 15:00 local "
          "time. Outside core hours, employees may arrange their schedule "
          "freely."),
    ("p", "All permanent employees receive 27 days of paid annual leave per "
          "year, rising to 32 days after five years of continuous service. "
          "Leave requests longer than ten consecutive days require approval "
          "from a team lead at least six weeks in advance. Unused leave may be "
          "carried into the following year up to a maximum of 5 days."),
    ("p", "Parental leave is offered for 46 weeks at full pay, shared between "
          "partners as they choose. Northvale additionally offers a "
          "\"settling week\" — five paid days taken in the month a new "
          "employee joins, intended for relocation and administrative tasks."),

    ("pagebreak", ""),

    ("h2", "3. Remote work policy"),
    ("p", "Employees may work remotely up to three days per week. Fully remote "
          "arrangements are possible but must be approved by both the team "
          "lead and the Head of People, and are reviewed every twelve months. "
          "Staff working remotely from outside Norway, Portugal or Canada are "
          "limited to 60 days per calendar year for tax reasons."),
    ("p", "Northvale provides a one-time home office allowance of 9,000 NOK, "
          "claimable within the first six months of employment. It covers a "
          "desk, chair and monitor. Laptops and phones are issued separately "
          "by the IT team and are not part of this allowance."),

    ("h2", "4. Engineering practices"),
    ("p", "All production code requires review by at least two engineers "
          "before merging. Deployments to production are frozen between 15 "
          "December and 6 January, and on any Friday after 14:00. The on-call "
          "rotation covers a full week and is compensated with two additional "
          "days of leave per rotation."),
    ("p", "Tidewater runs on a cluster of 64 compute nodes hosted in Oslo. "
          "Model retraining happens every Sunday at 02:00 UTC and takes "
          "approximately nine hours. If retraining fails twice consecutively, "
          "the previous model version is retained automatically and the "
          "on-call engineer is paged."),

    ("h2", "5. Company timeline"),
    ("p", "March 2019 — Northvale Analytics is founded in Bergen with four "
          "employees and a grant of 2.4 million NOK from the Norwegian "
          "Research Council."),
    ("p", "September 2020 — The first version of Tidewater is deployed with a "
          "single pilot customer, Vestkraft."),
    ("p", "June 2022 — The Lisbon office opens with eleven employees, focused "
          "on customer support and data operations."),
    ("p", "February 2024 — Northvale raises 180 million NOK in Series B "
          "funding, led by Halden Ventures."),
    ("p", "August 2025 — The Montreal office opens, marking the company's "
          "first presence outside Europe."),
    ("p", "January 2026 — Tidewater version 4 is released, extending the "
          "forecast horizon from seven days to fourteen."),

    ("pagebreak", ""),

    ("h2", "6. Equipment and expenses"),
    ("p", "Every employee receives a laptop on their first day. Engineers "
          "receive a machine with 32 GB of memory; all other roles receive 16 "
          "GB. Laptops are replaced every three years, or earlier if a fault "
          "cannot be repaired within five working days."),
    ("p", "Travel expenses are reimbursed within 21 days of submission. "
          "Receipts are required for any single expense above 300 NOK. Train "
          "travel is preferred over air travel for journeys under six hours, "
          "and Northvale will cover the cost difference where train tickets "
          "are more expensive."),

    ("h2", "7. Learning budget"),
    ("p", "Each employee has an annual learning budget of 18,000 NOK, usable "
          "for conferences, courses and books. The budget does not carry over "
          "between years. Conference attendance additionally grants up to "
          "three days of paid study leave, separate from annual leave."),
    ("p", "Northvale runs an internal seminar series every second Thursday, "
          "where employees present work in progress. Attendance is optional "
          "but presenting once per year is expected of all engineering staff."),

    ("h2", "8. Security and access"),
    ("p", "All employees must use hardware security keys for authentication. "
          "Two keys are issued to each person: a primary key kept on a "
          "keyring and a backup key stored at home. Lost keys must be "
          "reported to the security team within four hours, after which the "
          "key is revoked and a replacement is dispatched the same day."),
    ("p", "Access to customer data is granted on a per-project basis and "
          "expires automatically after 90 days unless renewed. Only members "
          "of the data operations team may export raw consumption data, and "
          "every export is logged with the requester's name, the volume of "
          "records and the stated purpose. Northvale retains these logs for "
          "seven years."),
    ("p", "Production databases may not be accessed from personal devices "
          "under any circumstances. Engineers who need production access "
          "during an incident use a dedicated jump host in Oslo, which "
          "records all sessions. Recordings are kept for 30 days and reviewed "
          "monthly by the Head of Security, currently Anneli Fossum."),

    ("pagebreak", ""),

    ("h2", "9. Customers and support"),
    ("p", "Northvale serves eleven grid operators. The four largest by "
          "contract value are Vestkraft in Norway, Iberdata Energia in "
          "Portugal, Nordlys Nett in Norway and Baltic Grid Services in "
          "Estonia. Together these four account for roughly 68 percent of "
          "annual recurring revenue."),
    ("p", "Support operates from the Lisbon office between 07:00 and 19:00 "
          "Central European Time on working days. Outside those hours, "
          "critical incidents are handled by the on-call engineer in Bergen. "
          "A critical incident is defined as any failure that prevents a "
          "customer from retrieving a forecast for more than 30 minutes."),
    ("p", "The contractual response time for critical incidents is 15 "
          "minutes, and for non-critical issues it is one working day. "
          "Northvale has met the critical target in 99.2 percent of cases "
          "since the current agreement took effect in April 2024."),

    ("h2", "10. How Tidewater works"),
    ("p", "Tidewater combines three data sources. The first is numerical "
          "weather prediction, refreshed every six hours, which provides "
          "temperature, wind speed and cloud cover on a ten-kilometre grid. "
          "The second is historical consumption, supplied by each customer at "
          "hourly resolution and typically covering at least four years. The "
          "third is an industrial calendar recording planned shutdowns at "
          "large facilities, which the customer maintains manually."),
    ("p", "The model itself is a gradient-boosted ensemble with a separate "
          "component per forecast horizon. Version 4 introduced a correction "
          "layer that adjusts predictions during public holidays, which "
          "reduced mean absolute error on those days by 23 percent compared "
          "with version 3."),
    ("p", "Forecast accuracy is reported to customers each month. Across all "
          "customers in 2025, the mean absolute percentage error was 3.1 "
          "percent at 24 hours ahead and 7.8 percent at seven days ahead. "
          "Accuracy at fourteen days has not yet been reported, as version 4 "
          "was released too recently to accumulate a full month of data."),

    ("h2", "11. Hiring process"),
    ("p", "Applications are reviewed within ten working days. Candidates who "
          "progress complete a take-home exercise expected to take no more "
          "than four hours, followed by two interviews: a technical "
          "discussion of the exercise, and a conversation with the team they "
          "would join."),
    ("p", "Northvale does not use whiteboard coding interviews. Offers are "
          "made within five working days of the final interview and remain "
          "open for two weeks. Every candidate who completes the take-home "
          "exercise receives written feedback, whether or not an offer is "
          "made."),

    ("pagebreak", ""),

    ("h2", "12. Frequently asked questions"),
    ("p", "Can I take my laptop abroad? Yes, within the European Economic "
          "Area and Canada without prior approval. For other destinations, "
          "notify the IT team at least one week in advance so that device "
          "encryption settings can be checked."),
    ("p", "What happens to the learning budget if I join mid-year? It is "
          "prorated by the number of full months remaining in the calendar "
          "year, rounded up."),
    ("p", "Is the settling week paid separately from annual leave? Yes. The "
          "five settling days do not count against the 27-day annual leave "
          "entitlement."),
    ("p", "Who approves fully remote arrangements? Both the team lead and the "
          "Head of People must approve, and the arrangement is reviewed every "
          "twelve months."),

    ("h2", "13. Onboarding"),
    ("p", "New employees follow a structured four-week onboarding. In week "
          "one, no production work is assigned; the time is spent on the "
          "settling week, equipment setup and reading the internal "
          "documentation. Week two introduces the codebase through a guided "
          "exercise: every new engineer ships a small, real change to "
          "Tidewater, reviewed by their assigned mentor."),
    ("p", "Each new employee is assigned two people: a mentor, who answers "
          "technical questions and reviews early work, and a buddy from a "
          "different team, whose role is purely social and lasts three "
          "months. Mentors are never the new employee's team lead, so that "
          "questions can be asked without any bearing on evaluation."),
    ("p", "At the end of week four, the new employee writes a short document "
          "listing everything they found confusing during onboarding. These "
          "documents are read by the whole team and are the primary source of "
          "improvements to internal documentation. Northvale considers this "
          "the most valuable artefact a new joiner produces, because the "
          "ability to see what is confusing disappears within a month."),

    ("h2", "14. Performance and progression"),
    ("p", "Performance conversations happen twice a year, in April and "
          "October. They are explicitly separated from salary reviews, which "
          "take place once a year in February. Northvale separates the two "
          "deliberately, on the view that a conversation about growth is "
          "distorted when money is decided in the same meeting."),
    ("p", "The engineering ladder has five levels: Engineer I, Engineer II, "
          "Senior Engineer, Staff Engineer and Principal Engineer. Promotion "
          "to Senior Engineer or above requires a written case prepared "
          "jointly by the employee and their team lead, reviewed by a panel "
          "of three people from outside the team."),
    ("p", "There is no fixed time in level and no quota on promotions per "
          "cycle. The median time from Engineer II to Senior Engineer at "
          "Northvale has been 26 months."),

    ("pagebreak", ""),

    ("h2", "15. Offices and facilities"),
    ("p", "The Bergen headquarters occupies two floors of a converted "
          "warehouse on Bryggen, with desk space for 95 people and four "
          "meeting rooms. The building has a bicycle store with 40 spaces and "
          "showers on the ground floor. Northvale reimburses public transport "
          "passes in full for all employees in all three offices."),
    ("p", "The Lisbon office houses 38 people in Alcântara and operates on a "
          "hot-desking basis, since most of the support team works partly "
          "remotely. The Montreal office, opened in August 2025, currently "
          "has 15 employees and is expected to reach 25 by the end of 2026."),
    ("p", "Lunch is provided free in Bergen and Lisbon. In Montreal, where "
          "the office has no kitchen, employees receive a daily allowance of "
          "18 Canadian dollars instead. All three offices are closed on local "
          "public holidays, and employees may swap any public holiday for a "
          "different day if it does not suit them."),

    ("h2", "16. Environmental policy"),
    ("p", "Northvale committed in 2023 to reducing emissions per employee by "
          "half by 2030, measured against a 2022 baseline. The largest single "
          "source is air travel, which is why train travel is mandatory for "
          "journeys under six hours. In 2025, air travel accounted for 61 "
          "percent of measured emissions, down from 74 percent in 2022."),
    ("p", "The Oslo compute cluster runs on a contract specifying 100 percent "
          "renewable electricity. Model retraining is deliberately scheduled "
          "for Sunday nights, when grid demand and carbon intensity are "
          "lowest. Northvale publishes an emissions report each March."),

    ("h2", "17. Data retention"),
    ("p", "Customer consumption data is retained for the duration of the "
          "contract plus 24 months, after which it is deleted automatically. "
          "Forecasts produced by Tidewater are retained for five years, since "
          "they are needed to evaluate long-term model accuracy."),
    ("p", "Internal communications are retained for three years. Access logs "
          "are retained for seven years, and jump host session recordings for "
          "30 days. Employees may request deletion of their personal data on "
          "leaving, with the exception of records Northvale is legally "
          "required to keep, such as payroll documents."),

    ("h2", "18. Contact"),
    ("p", "Questions about this handbook should be directed to the People "
          "team at people@northvale.example. The handbook is reviewed each "
          "January and was last updated on 12 January 2026. The current Head "
          "of People is Marek Dubois, based in the Bergen office."),
]


def build():
    OUT.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=2.2 * cm, bottomMargin=2.2 * cm,
        title="Northvale Analytics Handbook 2026",
    )
    story = []
    for kind, text in S:
        if kind == "h1":
            story += [Paragraph(text, h1), Spacer(1, 6)]
        elif kind == "h2":
            story.append(Paragraph(text, h2))
        elif kind == "pagebreak":
            story.append(PageBreak())
        else:
            story.append(Paragraph(text, body))
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
