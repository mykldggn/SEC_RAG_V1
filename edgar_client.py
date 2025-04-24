"""
Download and parse SEC filings via HTML scraping of EDGAR browse & index pages.
"""
import requests
from bs4 import BeautifulSoup
from .config import EDGAR_USER_AGENT
class EdgarClient:
    def __init__(self, user_agent: str = None):
        self.user_agent = user_agent or EDGAR_USER_AGENT
        self.base_url = "https://www.sec.gov"

    def fetch_filings(self, cik: str, filing_types: list[str]) -> list[dict]:
        """
        For each ftype in filing_types, hit the browse-edgar page (count=100),
        parse the filings table, then for each row:
          1) GET the '-index.html' page
          2) scrape its <table class="tableFile"> first row's link
             (which points to the real filing HTML)
        Returns:
          [
            {
              "cik": cik,
              "filingType": dtype,        # e.g. "10-K" or "10-K/A"
              "documentId": accession,    # accession-number
              "date": date,
              "htmlUrl": full_doc_url     # direct HTML of filing
            },
            ...
          ]
        """
        headers = {"User-Agent": self.user_agent}
        padded = f"{int(cik):010d}"
        records: list[dict] = []

        for ftype in filing_types:
            browse_url = (
                f"{self.base_url}/cgi-bin/browse-edgar?"
                f"action=getcompany&CIK={padded}"
                f"&type={ftype}&count=100&owner=exclude"
            )
            bresp = requests.get(browse_url, headers=headers)
            bresp.raise_for_status()
            bsoup = BeautifulSoup(bresp.text, "html.parser")

            table = bsoup.find("table", class_="tableFile2")
            if not table:
                continue

            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue
                dtype = cols[0].get_text(strip=True)
                if not dtype.upper().startswith(ftype.upper()):
                    continue

                link = cols[1].find("a")["href"]               # e.g. "/Archives/.../0000789019-25-000049-index.html"
                date = cols[3].get_text(strip=True)
                accession = link.split("/")[-1].replace("-index.html", "")

                index_url = self.base_url + link
                iresp = requests.get(index_url, headers=headers)
                iresp.raise_for_status()
                isoup = BeautifulSoup(iresp.text, "html.parser")

                ftable = isoup.find("table", class_="tableFile")
                if not ftable:
                    continue

                doc_row = ftable.find_all("tr")[1]
                doc_href = doc_row.find("a")["href"]
                doc_url  = self.base_url + doc_href

                records.append({
                    "cik":        cik,
                    "filingType": dtype,
                    "documentId": accession,
                    "date":       date,
                    "htmlUrl":    doc_url
                })

        return records