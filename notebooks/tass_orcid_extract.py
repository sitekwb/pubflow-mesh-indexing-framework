import xml.etree.ElementTree as ET
import requests
from tqdm import tqdm

with open('./out/pubmed_no_aff.csv') as f:
    with open('./out/pubmed_from_orcid.csv', 'w') as fw:
        fw.write(f"pmid;orcid;aff_id_from_orcid;year;country_from_orcid\n")
        for line in tqdm(f, total=666713):
            try:
                pmid, orcid, year, affiliation_id, country_cleaned = line.split(';')
                response = requests.get(f'https://pub.orcid.org/v3.0/{orcid}/activities')
                root = ET.fromstring(response.content)
                org_id = root.find('.//{http://www.orcid.org/ns/common}disambiguated-organization-identifier').text
                country = root.find('.//{http://www.orcid.org/ns/common}country').text
                fw.write(f"{str(pmid)};{orcid};{org_id};{year};{country}\n")
            except KeyboardInterrupt:
                break
            except:
                continue
