import json
import re
import sys
from datetime import datetime
from itertools import groupby
import requests
from dateparser.search import search_dates
from dateutil import parser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rapidfuzz import fuzz
from tqdm import tqdm

with open('config.json', 'r') as f:
    config_data = json.load(f)

url = config_data["vllm_endpoint"]
headers = {
    "Content-Type": "application/json",
}


def log_exception(func_name, logfile):
    exc_type, exc_obj, tb = sys.exc_info()
    lineno = tb.tb_lineno
    error_message = f"\n{datetime.now()} In {func_name} LINE.NO-{lineno} : {exc_obj}"
    print("error_message : ", error_message)
    with open(logfile, 'a', encoding='utf-8') as fp:
        fp.writelines(error_message)


def process_logger(process, logfile):
    with open(logfile, 'a', encoding='utf-8') as fp:
        fp.writelines(f'\n{datetime.now()} {process}')


def re_encounter(splitDate, encounter_name, context, page_num, vllm_endpoint, logfile):
    try:
        seDates = []
        stop_words = list(set(stopwords.words('english')))
        for sp in stop_words:
            splitDate = re.sub(fr"\b{sp}\b", "", splitDate)
        regSp = r"[~!@#$%^&*()_+=`{[\]};:'\",<>\|?]"
        splitDate = re.sub("\s+", " ", re.sub(regSp, " ", splitDate))

        seDates = search_dates(splitDate, settings={'STRICT_PARSING': True, 'DATE_ORDER': 'MDY'})
        enc = []
        if seDates:
            ftDate = []
            # IDENTIFY DATES RANGE
            for index, data in enumerate(seDates):
                if bool(re.search("[^A-Za-z\s+0-9]", data[0])):
                    # if len((data[0]).split()) > 1:
                    #     for le in data[0].split():
                    #         leDate = search_dates(le, settings={'STRICT_PARSING': True, 'DATE_ORDER': 'MDY'})
                    #         if leDate:
                    #             inxDate = [{"start": m.start(0), "end": m.end(0)} for m in
                    #                        re.finditer(str(le).lower(), splitDate.lower())]
                    #             ftDate.extend(inxDate)
                    #             break
                    # else:
                    inxDate = [{"start": m.start(0), "end": m.end(0)} for m in
                               re.finditer(str(data[0]).lower(), splitDate.lower())]
                    ftDate.extend(inxDate)
                elif [True for ls in data[0].split() if ls.isalpha()] and [True for ls in data[0].split() if
                                                                           ls.isdigit()]:
                    inxDate = [{"start": m.start(0), "end": m.end(0)} for m in
                               re.finditer(str(data[0]).lower(), splitDate.lower())]
                    ftDate.extend(inxDate)
            # GET END OF DATE
            endIndex = {}
            for index, data in enumerate(ftDate):
                end = data['end'] + 1
                if index < len(ftDate) - 1:
                    if not end == ftDate[index + 1]['start']:
                        endIndex = data
                        break
                else:
                    endIndex = data
            if endIndex:
                splitDateRe = splitDate[:endIndex['end']]
                seDates = []
                seDates = search_dates(splitDateRe, settings={'STRICT_PARSING': True, 'DATE_ORDER': 'MDY'})
                # CHECK DATES LEN
                if seDates:
                    if len(seDates) > 1:
                        if len(seDates) == 2:
                            if seDates[0][1].strftime("%Y-%m-%d") == seDates[1][1].strftime("%Y-%m-%d"):
                                dd = seDates[0][1].strftime("%Y-%m-%d")
                                pDate = parser.parse(dd)
                                if datetime.now().year + 4 > pDate.year:
                                    enc.append(dd)
                            else:
                                alpDate = []
                                for data in seDates:
                                    if not "".join(data[0].split()).isalpha():
                                        dd = data[1].strftime("%Y-%m-%d")
                                        pDate = parser.parse(dd)
                                        if datetime.now().year + 4 > pDate.year:
                                            alpDate.append(dd)
                                enc.extend(alpDate)
                        else:
                            enc.append("")
                    else:
                        dd = seDates[0][1].strftime("%Y-%m-%d")
                        pDate = parser.parse(dd)
                        if datetime.now().year + 4 > pDate.year:
                            enc.append(dd)
                if not enc:
                    encounter_date = toc_encounter_date_vllm(splitDateRe, encounter_name, page_num, vllm_endpoint,
                                                             logfile)
                    enc.append(encounter_date)
        return enc
    except Exception as e:
        enc = []
        encounter_date = toc_encounter_date_vllm(context, encounter_name, page_num, vllm_endpoint, logfile)
        if not encounter_date:
            log_exception("re_encounter", logfile)
        else:
            enc.append(encounter_date)
        return enc


def count_tokens(text):
    tokens = word_tokenize(text)
    return len(tokens)


def vllmAPI(content, context, task, vllm_endpoint, logfile):
    try:
        url = vllm_endpoint
        input_prompt = f"Context : {context} \n\n task : {task}\n Answer:"
        tokenlength = count_tokens(context)
        if tokenlength > 7000:
            context_length = int(tokenlength + (0.4 * tokenlength))
        else:
            context_length = 8192
        payload = {
            "messages": [
                {"role": "system",
                 "content": content},
                {"role": "user", "content": input_prompt},
            ],
            "model": config_data["model_name"],
            "stream": False,
            "max_tokens": context_length,
            "stop": None,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "temperature": 0,
            "top_p": 0.95
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        result = response.json()
        if str(response.status_code) != "200":
            process_logger(f"{result}", logfile)
        content = result['choices'][0]['message']['content']
        return content
    except Exception as e:
        log_exception("vllmAPI", logfile)
        return ""


def chunk_texts(all_text, ranges, logfile):
    try:
        OV_data_pages = []
        for i, rang in enumerate(ranges):
            st, en = int(rang.split('-')[0]), int(rang.split('-')[1])
            if i == 0 and not st == 1:
                ranges = f"{1}-{st - 1}"
                OV_data_pages.append({
                    "sno": int(ranges.split("-")[0]),
                    "page_range": ranges
                })
            if i == len(all_text) - 1 and not en == len(all_text):
                ranges = f"{en}-{len(all_text)}"
                OV_data_pages.append({
                    "sno": int(ranges.split("-")[0]),
                    "page_range": ranges
                })

            ranges = f"{st}-{en}"
            OV_data_pages.append({
                "sno": int(ranges.split("-")[0]),
                "page_range": ranges
            })
        OV_data_pages = sorted(OV_data_pages, key=lambda x: x['sno'])
        for data in OV_data_pages:
            del data["sno"]
        return OV_data_pages
    except Exception as e:
        log_exception("chunk_texts:", logfile)
        return []


def toc_encounter_date_vllm1(context, encounter_name, page_num, vllm_endpoint, logfile):
    try:
        if encounter_name:
            task = """"Extract given """ + encounter_name + """" only from the context.
                    If Dates doesn't have return empty string.
                    Only rely on the information explicitly present in the given context. Do not use external knowledge.
                    Extract Proper or Valid Date only from given context.
                    Return YYYY/mm/dd format.
                    Ensure that your reply follows the following valid JSON format with NO PREAMBLE and POSTAMBLE:
                    [{
                        "date":""
                    }]
                    """
            content = f"You are an expert in extracting {encounter_name} from medicine reports. Your task is to retrieve data from the provided context according to the given instructions."

            # else:
            #     task = """"Extract only mentioned in this date of service , dos, visit date,visited on, signed by,office visit ,admission date,collection or collected date into "date" params from the context.
            #                 If Dates doesn't have return empty string.
            #                 Only rely on the information explicitly present in the given context. Do not use external knowledge.
            #                 Extract Proper or Valid Date only from given context.
            #                 Return YYYY/mm/dd format.
            #                 Ensure that your reply follows the following valid JSON format with NO PREAMBLE and POSTAMBLE:
            #                 [{
            #                     "date":""
            #                 }]
            #             """
            #     content = f"You are an expert in extracting date of service,visit date,visited on, office visit from medicine reports. Your task is to retrieve data from the provided context according to the given instructions."

            en_date = vllmAPI(content, context.lower(), task, vllm_endpoint, logfile)

            # EXTRACTION FROM   JSON
            if en_date:
                if "[" not in en_date and "]" not in en_date:
                    en_date = f"[{en_date}]"
            else:
                en_date = "1900-01-01"
            reg_pattern = r'\[\s*{.*?}\s*\]'
            textre = re.findall(reg_pattern, en_date, re.DOTALL)
            encounter_list = []
            if textre:
                encounter_json = json.loads(textre[0])
                splitDate = encounter_json[len(encounter_json) - 1]["date"]
                seDates = search_dates(splitDate,
                                       settings={'STRICT_PARSING': True, 'DATE_ORDER': 'MDY'})
                if seDates:
                    dateRe = seDates[0][1].strftime("%Y-%m-%d")
                    dt = parser.parse(dateRe)
                    if 2000 < dt.year < datetime.now().year + 4:
                        encounter_list.append(dateRe)
            if encounter_list:
                encounter_date = encounter_list[0]
            else:
                encounter_date = ""
        else:
            encounter_date = ""
        return encounter_date
    except Exception as e:
        log_exception("toc_encounter_date_vllm", logfile)
        return ""


def get_res_doc_fac_pro_toc(context, vllm_endpoint, logfile):
    dict_ = {"DOS": "",
             "DOB": "", }
    try:
        instruction = "Identifying the presence of DOS , DOB."
        task = """Analyse the prompt and given context,if requirement is satisfied for prompt and context,Extract the DOS , DOB from the provided context or else empty string.
                General Rules:
                    - Extract only the information related to DOS , DOB present in the context.
                    - Only the contents under DOS , DOB should be considered as appointment details reports.
                    - If the appointment date has Expiration Date neglect them.

                Classification Rules:
                    - If multiple DOS , DOB are available then extract them one by one and return each as separate JSON Object.
                    - Appointment date and DOS is not the same date.
                "DOS" : "Return the current date of Patient visit or service or encounter or assessments. If no such date is given return `YYYY-MM-DD`.",
                "DOB" : "date of birth (DOB) for patient.If no such date is given return `YYYY-MM-DD`"
                Formatting Instructions:
                    - Do not add any preamble or postamble text to the final output.
                Output  Constraints:
                    {

                        "DOS":"",
                        "DOB":"",

                    }
                    """
        response = vllmAPI(instruction, context, task, vllm_endpoint, logfile)
        # EXTRACTION FROM JSON
        reg_pattern = r'\{[\s\S]*?\}'

        textre = re.findall(reg_pattern, str(response), re.DOTALL)

        if textre:
            try:
                encounter_json = json.loads(textre[0].strip())
                dos = encounter_json["DOS"] if "DOS" in encounter_json else ""
                dob = encounter_json["DOB"] if "DOB" in encounter_json else ""
                if dos:
                    if dos == "YYYY-MM-DD":
                        dos = ""
                    elif dos == dob:
                        dos = ""
                if dos:
                    seDates = search_dates(dos,
                                           settings={'STRICT_PARSING': True, 'DATE_ORDER': 'MDY'})

                    if seDates:
                        dateRe = seDates[0][1].strftime("%Y-%m-%d")
                        dt = parser.parse(dateRe)
                        if 2000 < dt.year < datetime.now().year + 4:
                            dos = dateRe
                dict_["DOS"] = dos
                dict_["DOB"] = dob
            except Exception as e:
                log_exception("get_res_doc_fac_pro_toc load:", logfile)

        return dict_
    except Exception as e:
        log_exception("get_res_doc_fac_pro_toc", logfile)
        return {"DOS": "",
                "DOB": "", }


def check_toc_range(context, vllm_endpoint, logfile):
    try:
        instructions = "You are an expert in Identifying the presence of start or end of visit in the given context,your task is to analyze and classify the provided text as per the given instructions"
        task = """ Analyze the given clinical note and classify whether it represents the start, ongoing, or end of a patient visit.  
                - "start of visit": Text describing the beginning of an encounter such as visit reason, chief complaint, intake or triage notes, "Patient seen for...", "Presents with...", or similar language clearly marking the start.  
                - "ongoing visit": Text focused on the middle portion of care such as history of present illness (HPI), examination, assessments, labs, or progress updates without clear start or discharge cues.  
                - "end of visit": Text describing completion of the visit such as plan, discharge instructions, follow-up, referrals, medications prescribed, procedures ordered, appointment scheduling, or provider sign-off.  
                - If appointment or follow-up information is present, classify as "end of visit".  
                Do not infer beyond explicit text; ignore vague or unrelated phrases.  
                Return valid JSON only, with no extra text or explanation, in the exact format below:          

                ### Output Constraints:
                - **Do not include the provided text in the output.**
                - **Ensure the output strictly follows this JSON format:**
                - **Do not add any preamble or postamble text to the final output**.
                ```json
                {
                    "report_type": <"start of visit" or "ongoing visit" or "end of visit" or "">}
                }"""

        reg = r"[^A-Za-z0-9]"
        result = vllmAPI(instructions, context, task, vllm_endpoint, logfile)
        try:
            match = re.search(r'\{[\s\S]*}', result, re.DOTALL)
            if not match:
                return ""
            extracted_json = match.group(0)

            # Load and dump to ensure correct formatting
            cleaned_json = json.loads(extracted_json)

            # 🔑 THIS IS THE KEY PART
            if isinstance(cleaned_json, str):
                cleaned_json = json.loads(cleaned_json)

            if not isinstance(cleaned_json, dict):
                return ""

            report_type = cleaned_json.get("report_type", "")
            report_type = re.sub(r'[^A-Za-z ]', '', report_type).strip()
            return report_type

        except Exception as e:
            log_exception(f"check_Report_type_load error:", logfile)
            match = re.search(r'\{[\S\s]*}', result, re.DOTALL)
            if match:
                extracted_json = match.group(0)
                # Load and dump to ensure correct formatting
                try:
                    cleaned_json = json.loads(extracted_json)
                    report_type = cleaned_json["report_type"]
                    report_type = re.sub(reg, "", report_type).strip()
                    return report_type
                except Exception as e:
                    log_exception(f"check_Report_type_load error{extracted_json}", logfile)
                    return ""
            else:
                return ""

    except Exception as e:
        log_exception("check_report_type", logfile)
        return ""


def toc_encounter_date_vllm(context, encounter_name, page_num, vllm_endpoint, logfile):
    try:
        if encounter_name:
            task = """"Extract given """ + encounter_name + """" only from the context.
                    If Dates doesn't have return empty string.
                    Only rely on the information explicitly present in the given context. Do not use external knowledge.
                    Extract Proper or Valid Date only from given context.
                    Return YYYY/mm/dd format.
                    Return only single DOS date for given """ + encounter_name + """.
                    Ensure that your reply follows the following valid JSON format with NO PREAMBLE and POSTAMBLE:
                    {
                        "date":""
                    }
                    """
            content = f"You are an expert in extracting {encounter_name} from medicine reports. Your task is to retrieve data from the provided context according to the given instructions."
        else:
            task = """"Extract only mentioned in this date of service , dos, visit date,Visit Note,visited on, signed by,office visit , RP Date ID, admission date,collection or collected date into "date" params from the context.
                    If Dates doesn't have return empty string.
                    Only rely on the information explicitly present in the given context. Do not use external knowledge.
                    Extract Proper or Valid Date only from given context.
                    Return YYYY/mm/dd format.
                    Return only single DOS date.
                    Ensure that your reply follows the following valid JSON format with NO PREAMBLE and POSTAMBLE:
                   {
                       "date":""
                   }
                    """
            content = f"You are an expert in extracting date from medicine reports. Your task is to retrieve data from the provided context according to the given instructions."

        en_date = vllmAPI(content, context.lower(), task, vllm_endpoint, logfile)
        encounter_date = ""
        try:
            matches = re.findall(r"\{[\s\S]*?\}", en_date)

            if not matches:
                return ""

            for block in matches:
                try:
                    obj = json.loads(block)
                    if isinstance(obj, str):
                        obj = json.loads(obj)

                    if not isinstance(obj, dict):
                        continue

                    splitDate = obj.get("date", "")
                    if not splitDate:
                        continue

                    seDates = search_dates(
                        splitDate,
                        settings={'STRICT_PARSING': True, 'DATE_ORDER': 'MDY'}
                    )

                    if not seDates:
                        continue

                    dateRe = seDates[0][1].strftime("%Y-%m-%d")
                    dt = parser.parse(dateRe)

                    if 2000 < dt.year < datetime.now().year + 4:
                        encounter_date = dateRe
                        break

                except Exception:
                    continue
        except Exception as e:
            # EXTRACTION FROM JSON
            reg_pattern = r'\{[\S\s]*}'

            textre = re.findall(reg_pattern, en_date, re.DOTALL)
            encounter_date = ""
            if textre:
                try:
                    encounter_json = json.loads(textre[0])
                    splitDate = encounter_json["date"] if "date" in encounter_json else ""
                    seDates = search_dates(splitDate,
                                           settings={'STRICT_PARSING': True, 'DATE_ORDER': 'MDY'})
                    if seDates:
                        dateRe = seDates[0][1].strftime("%Y-%m-%d")
                        dt = parser.parse(dateRe)
                        if 2000 < dt.year < datetime.now().year + 4:
                            encounter_date = dateRe
                except Exception as e:
                    log_exception(f"toc_encounter_date loads_first:{en_date}", logfile)
                    try:
                        textre = re.findall(r"{.*?}", en_date, re.DOTALL)
                        if textre:
                            encounter_json = json.loads(textre[0])
                            splitDate = encounter_json["date"] if "date" in encounter_json else ""
                            seDates = search_dates(splitDate,
                                                   settings={'STRICT_PARSING': True, 'DATE_ORDER': 'MDY'})
                            if seDates:
                                dateRe = seDates[0][1].strftime("%Y-%m-%d")
                                dt = parser.parse(dateRe)
                                if 2000 < dt.year < datetime.now().year + 4:
                                    encounter_date = dateRe
                    except Exception as e:
                        log_exception(f"toc_encounter_date loads_sec:{en_date}", logfile)

        return encounter_date

    except Exception:
        log_exception("toc_encounter_date_vllm", logfile)
        return ""


def toc_encounter_date1(all_text, start, end, enc_li, vllm_endpoint, logfile):
    try:
        reg = r"[^A-Za-z0-9]"
        context = "\n".join(all_text[start:end])
        context = context.replace('"', '')
        page_num = f'{start + 1}-{end}'
        if enc_li:
            indexEn = []
            encounter_name = ""
            textsp = ""
            for encounter_name_ in enc_li:
                textsp = context.replace("@", " ").replace("¢", " ").replace("‘", "").replace("Â¢",
                                                                                              " ").replace(
                    "â€˜", "")
                textsp = textsp.lower()
                textsp1 = re.sub("\s+", " ", "".join(textsp.replace(f"{encounter_name_}:", encounter_name_)))
                textsp1 = re.sub("\s+", " ", "".join(textsp1.replace(f"{encounter_name_}.", encounter_name_)))
                if not textsp1.strip():
                    textsp = re.sub("\s+", " ", "".join(textsp.replace(f"{encounter_name_}:", encounter_name_)))
                    textsp = re.sub("\s+", " ", "".join(textsp.replace(f"{encounter_name_}.", encounter_name_)))
                else:
                    textsp = textsp1

                indexEn = [{"start": m.start(0), "end": m.end(0)} for m in re.finditer(encounter_name_, textsp)]
                if not indexEn:
                    indexEn = [{"start": m.start(0), "end": m.end(0)} for m in re.finditer(encounter_name_, textsp)]
                indexEn = sorted(indexEn, key=lambda x: x["start"])
                if indexEn:
                    encounter_name = encounter_name_
                    break

            if indexEn:
                splitContext = ""
                enc = []
                for en in indexEn:
                    splitDate = textsp[en["start"]:en["end"] + 150]
                    splitContext += splitDate
                enc.extend(re_encounter(splitContext, encounter_name, textsp, page_num, vllm_endpoint, logfile))
                if not enc:
                    splitContext = ""
                    enc = []
                    for en in indexEn:
                        splitDate = textsp[en["start"] - 80:en["end"]]
                        splitContext += splitDate
                    enc.extend(re_encounter(splitContext, encounter_name, textsp, page_num, vllm_endpoint, logfile))

                if enc:
                    encounter_date = enc[0]
                    if encounter_date:
                        dt = parser.parse(encounter_date)
                        if dt.year < 2000:
                            encounter_date = toc_encounter_date_vllm(splitContext if splitContext else textsp,
                                                                     encounter_name,
                                                                     page_num, vllm_endpoint, logfile)
                else:
                    encounter_date = toc_encounter_date_vllm(textsp, encounter_name, page_num, vllm_endpoint, logfile)
                return encounter_date
            else:
                context_ = ""
                for page_ in range(start, end):
                    en_encounter_name = re.sub(reg, "", encounter_name.lower())
                    en_text = re.sub(reg, "", all_text[page_].lower())
                    if fuzz.partial_ratio(en_encounter_name, en_text) > 90:
                        context_ += all_text[page_]
                if context_:
                    context = context_
                encounter_date = toc_encounter_date_vllm(context, encounter_name, page_num, vllm_endpoint, logfile)
                return encounter_date
        else:
            date_list = []
            context = "\n".join(all_text[start:end])
            encounter_date = toc_encounter_date_vllm(context, "", page_num, vllm_endpoint, logfile)
            date_list.append(encounter_date)
            date_list = list(set(date_list))
            gp_date_list = []
            for key, gp in groupby(date_list, key=lambda x: x):
                gp_list = list(gp)
                gp_date_list.append({"len": len(gp_list), "list": gp_list})
            sorted(gp_date_list, key=lambda x: x["len"], reverse=True)
            encounter_date = gp_date_list[0]["list"][0]
        return encounter_date
    except Exception as e:
        log_exception("toc_encounter_date_vllm", logfile)
        return ""


def toc_encounter_date(context, page_num, encounter_name, dos_list, dos_list1, vllm_endpoint, logfile):
    try:

        encounter_date = ""
        # REPLACE '.' VALUE  IN DATES
        context = context.replace("\n", " ")
        textsp = context.replace("@", " ").replace("¢", " ").replace("‘", "").replace("Â¢",
                                                                                      " ").replace(
            "â€˜", "")
        textsp = textsp.lower()
        if encounter_name:
            textsp1 = re.sub("\s+", " ", "".join(textsp.replace(f"{encounter_name}:", encounter_name)))
            if not textsp1.strip():
                textsp = re.sub("\s+", " ", "".join(textsp.replace(f"{encounter_name}:", encounter_name)))
            else:
                textsp = textsp1

            indexEn = [{"start": m.start(0), "end": m.end(0)} for m in re.finditer(encounter_name, textsp)]
            if not indexEn:
                indexEn = [{"start": m.start(0), "end": m.end(0)} for m in re.finditer(encounter_name, textsp)]
        else:
            textsp1 = re.sub("\s+", " ",
                             "".join([textsp.replace(f"{dos}:", dos) for dos in dos_list1 if dos in textsp]))
            if not textsp1.strip():
                textsp = re.sub("\s+", " ",
                                "".join([textsp.replace(f"{dos}:", dos) for dos in dos_list if dos in textsp]))
            else:
                textsp = textsp1

            indexEn = [{"start": m.start(0), "end": m.end(0)} for dos in dos_list1 for m in re.finditer(dos, textsp)]
            if not indexEn:
                indexEn = [{"start": m.start(0), "end": m.end(0)} for dos in dos_list for m in re.finditer(dos, textsp)]

            indexEn = sorted(indexEn, key=lambda x: x["start"])
            # CHECK THE ENCOUNTER NAME
            for dos in dos_list1:
                if dos in textsp:
                    encounter_name = dos

        indexEn = sorted(indexEn, key=lambda x: x["start"])
        # # CHECK THE ENCOUNTER NAME
        # for dos in dos_list1:
        #     if dos in textsp:
        #         encounter_name = dos

        indexEn = list(filter(None, indexEn))

        if indexEn:
            splitContext = ""
            enc = []
            for en in indexEn:
                splitDate = textsp[en["start"]:en["end"] + 150]
                splitContext += splitDate
            enc.extend(re_encounter(splitContext, encounter_name, textsp, page_num, vllm_endpoint, logfile))
            if not enc:
                splitContext = ""
                enc = []
                for en in indexEn:
                    splitDate = textsp[en["start"] - 80:en["end"]]
                    splitContext += splitDate
                enc.extend(re_encounter(splitContext, encounter_name, textsp, page_num, vllm_endpoint, logfile))

            if enc:
                encounter_date = enc[len(enc) - 1]
                if encounter_date:
                    dt = parser.parse(encounter_date)
                    if dt.year < 2000:
                        encounter_date = toc_encounter_date_vllm(splitContext if splitContext else textsp,
                                                                 encounter_name,
                                                                 page_num, vllm_endpoint, logfile)
            else:
                encounter_date = toc_encounter_date_vllm(textsp, encounter_name, page_num, vllm_endpoint, logfile)
            return encounter_date
        else:
            encounter_date = toc_encounter_date_vllm(context, encounter_name, page_num, vllm_endpoint, logfile)
            return encounter_date
    except Exception as e:
        log_exception("toc_encounter_date", logfile)
        return ""


def check_in_btw_range(dos_range, all_text, vllm_endpoint, logfile):
    try:
        fin_dos_range = []
        for data in dos_range:
            splitRr = data["page_range"].split("-")
            enc_name = data["encounter_name"]
            start = int(splitRr[0]) - 1
            end = int(splitRr[1])
            if (end - start) >= 10 or not enc_name:
                new_dos_range = [start + 1]
                for page_num in tqdm(range(start, end)):
                    context = all_text[page_num]
                    report_APP = check_toc_range(context, vllm_endpoint, logfile)
                    if report_APP.lower() == "endofvisit":
                        new_dos_range.append(page_num + 1)

                new_dos_range.append(end)
                new_dos_range = list(set(new_dos_range))
                new_dos_range = sorted(new_dos_range, key=lambda x: x)
                for i, page in enumerate(new_dos_range):
                    if i == 0:
                        if i == len(new_dos_range) - 1:
                            fin_dos_range.append({"page_range": f'{page}-{page}',
                                                  "encounter_name": data["encounter_name"] if page == end or
                                                                                              page == start else ""})
                        else:
                            fin_dos_range.append({"page_range": f'{page}-{new_dos_range[i + 1] - 1}',
                                                  "encounter_name": data["encounter_name"] if new_dos_range[
                                                                                                  i + 1] - 1 == end or
                                                                                              new_dos_range[
                                                                                                  i + 1] - 1 == start else ""})

                    elif i == len(new_dos_range) - 1:
                        fin_dos_range.append({"page_range": f'{page}-{page}', "encounter_name": data[
                            "encounter_name"] if page == start or page == end else ""})
                    else:
                        fin_dos_range.append({"page_range": f'{page}-{new_dos_range[i + 1] - 1}',
                                              "encounter_name": data["encounter_name"] if new_dos_range[
                                                                                              i + 1] - 1 == end or
                                                                                          new_dos_range[
                                                                                              i + 1] - 1 == start else ""})
            else:
                fin_dos_range.append(data)
        all_dos_range = []
        for i, data in enumerate(fin_dos_range):
            if i == 0:
                splitRr = data["page_range"].split("-")
                start = int(splitRr[0])
                end = int(splitRr[1])
                if start == 1:
                    all_dos_range.append({"page_range": data["page_range"], "encounter_name": data["encounter_name"]})
                else:
                    all_dos_range.append({"page_range": f"1-{start - 1}", "encounter_name": data["encounter_name"]})
                    all_dos_range.append({"page_range": f"{start}-{end}", "encounter_name": data["encounter_name"]})

            elif i == len(fin_dos_range) - 1:
                splitRr = data["page_range"].split("-")
                start = int(splitRr[0])
                end = int(splitRr[1])
                if end == len(all_text):
                    all_dos_range.append({"page_range": data["page_range"], "encounter_name": data["encounter_name"]})
                else:
                    all_dos_range.append({"page_range": f"{start}-{end}", "encounter_name": data["encounter_name"]})
                    all_dos_range.append(
                        {"page_range": f"{end}-{len(all_text)}", "encounter_name": data["encounter_name"]})

            else:
                all_dos_range.append({"page_range": data["page_range"], "encounter_name": data["encounter_name"]})

        # CHECK IN BTW RANGE
        fin_dos_range = check_in_btw_range(all_dos_range, all_text, vllm_endpoint, logfile)
        return fin_dos_range
    except Exception as e:
        log_exception("check_in_btw_range", logfile)
        return dos_range


def predict_dos(all_text, dos_list, dos_list1, vllm_endpoint, logfile):
    try:
        reg = r'[^A-Za-z0-9]'
        enc_list = []
        # start_dos -> 0:start vs, 1:end vs,2:empty 3:start & end vs
        for page_num, context in enumerate(all_text):
            text = re.sub(r"\s+", " ", re.sub(reg, " ", context.lower()))
            # enc = next((enc for enc in dos_list1 if fuzz.partial_ratio(enc, text) > 90), "")
            # enc1 = next((enc for enc in dos_list if fuzz.partial_ratio(enc, text) > 90), "")
            enc = [enc for enc in dos_list1 if fuzz.partial_ratio(enc, text) > 90]
            enc1 = [enc for enc in dos_list if fuzz.partial_ratio(enc, text) > 90]
            if enc or enc1:
                enc_list.append(
                    {"page_num": page_num + 1, "encounter_name": enc1 if enc1 else enc, "start_dos": 1 if enc else 0})
            else:
                if page_num == 0 or page_num == len(all_text) - 1:
                    enc_list.append(
                        {"page_num": 1 if page_num == 0 else len(all_text), "encounter_name": [], "start_dos": 2})
        dos_range = []
        for i, data in enumerate(enc_list):
            isStart = data["start_dos"]
            page_num = data["page_num"]
            enc = data["encounter_name"]
            if i == 0:
                l = 0
            else:
                pre_isStart = enc_list[i - 1]["start_dos"]
                if pre_isStart == 1 and isStart == 0:
                    start = enc_list[i - 1]["page_num"]
                    end = page_num
                    enc = [enc_list[i - 1]["encounter_name"],enc]
                    start_dos = 3
                elif pre_isStart == 0 and isStart == 1:
                    start = enc_list[i - 1]["page_num"]
                    end = page_num
                    enc = []
                    start_dos = 0
                    if end - start == 1:
                        continue
                    else:
                        start += 1
                        end -= 1
                elif pre_isStart == 1 and isStart == 1:
                    start = enc_list[i - 1]["page_num"]
                    end = page_num - 1
                    enc = enc_list[i - 1]["encounter_name"]
                    start_dos = 1

                elif isStart == 1:
                    # FIRST DATA
                    if i == 1:
                        start = 1
                        end = page_num - 1
                        enc = []
                        start_dos = 0
                    else:
                        start = enc_list[i - 1]["page_num"]
                        end = page_num
                        start_dos = 1
                else:
                    if dos_range:
                        start = int((dos_range[-1]["page_range"]).split("-")[1])
                    else:
                        start = enc_list[i - 1]["page_num"]
                    if not start == 1:
                        start += 1
                    end = page_num
                    start_dos = 0
                dos_range.append(
                    {"page_range": f'{start}-{end}', "encounter_name": enc, "start_dos": start_dos if enc else 2})
        return dos_range
    except Exception as e:
        log_exception("predict_dos", logfile)
        return []


def predict_dos1(all_text, dos_list, dos_list1, vllm_endpoint, logfile):
    try:
        reg = r'[^A-Za-z0-9]'
        enc_list = []
        for page_num, context in enumerate(all_text):
            text = re.sub(r"\s+", " ", re.sub(reg, " ", context.lower()))
            enc = next((enc for enc in dos_list1 if fuzz.partial_ratio(enc, text) > 90), "")
            enc1 = next((enc for enc in dos_list if fuzz.partial_ratio(enc, text) > 90), "")
            if enc or enc1:
                enc_list.append({"page_num": page_num + 1, "encounter_name": enc if enc else enc1,
                                 "start_dos": 0 if enc1 else 1})

            else:
                if page_num == 0 or page_num == len(all_text) - 1:
                    enc_list.append({"page_num": 1 if page_num == 0 else len(all_text), "encounter_name": "",
                                     "start_dos": 2})

        dos_range = []
        for i, data in enumerate(enc_list):
            isStart = data["start_dos"]
            page_num = data["page_num"]
            enc = data["encounter_name"]
            if i == 0:
                l = 0
            else:
                start = enc_list[i - 1]["page_num"]
                if not start == 1:
                    start += 1
                end = page_num
                dos_range.append({"page_range": f'{start}-{end}', "encounter_name": enc})

        # CHECK IN BTW RANGE
        fin_dos_range = check_in_btw_range(dos_range, all_text, vllm_endpoint, logfile)
        return fin_dos_range

    except Exception as e:
        log_exception("predict_dos", logfile)
        return []


def toc_extraction(all_text, file,vllm_endpoint, logfile):
    try:
        dos_list1 = ["encounter date", "enc date", "encounter dos", "entry date", "visit date", "RP Date ID",
                     "Visit Note", "Date seen"]
        dos_list = ["electronic signature", "signed electronically by", "electronically signed by",
                    "electronically authenticated by", "electronically signed out",
                    "electronically cosigned by", "electronically verified by", "electronically signed off",
                    "encounter signed-off", "encounter signed off", "encounter signed by",
                    "electronically signed on","electronically signed", "document generated by"]
        predictRange = []
        dos_range = predict_dos(all_text, dos_list, dos_list1, vllm_endpoint, logfile)

        tocRangeList = []

        if predictRange:
            print(f'length of Predict Range : {len(predictRange)}')
        elif dos_range:
            print(f'length of dosRange : {len(dos_range)}')

        if len(predictRange) == 0 and len(dos_range) == 0:

            page_list = []
            # PREDICT END OF VISIT
            for page_num, context in enumerate(tqdm(all_text)):
                report_APP = check_toc_range(context, vllm_endpoint, logfile)
                if report_APP.lower() == "endofvisit":
                    page_list.append(page_num)
                elif page_num == 0 or page_num == len(all_text) - 1:
                    page_list.append(page_num + 1)
            page_list = sorted(page_list, key=lambda x: x)
            toc_page_list = []
            # SPLIT RANGE
            for i, dos in enumerate(page_list):
                if i == 0:
                    start = page_list[i]
                    if i == 0 and i == len(page_list) - 1:
                        end = page_list[i]
                    else:
                        end = page_list[i + 1]
                    toc_page_list.append({"page_range": f"{start}-{end}"})
                elif not i == len(page_list) - 1:
                    end = page_list[i + 1]
                    start = page_list[i] + 1
                    toc_page_list.append({"page_range": f"{start}-{end}"})
                elif i == len(page_list) - 1:
                    dosIndex = toc_page_list[len(toc_page_list) - 1]["page_range"].split("-")
                    start = dosIndex[0]
                    end = dosIndex[1]
                    if not int(end) == int(page_list[len(page_list) - 1]):
                        toc_page_list.append({"page_range": f"{end}-{page_list[len(page_list) - 1]}"})
            # PREDICT DOS
            for data in tqdm(toc_page_list):
                start = int((data["page_range"]).split("-")[0]) - 1
                end = int((data["page_range"]).split("-")[1])
                context = ""
                if end - start > 5:
                    context += "\n".join(all_text[start:start + 2])
                    context += "\n".join(all_text[end - 2:end])
                else:
                    context += "\n".join(all_text[start:end])
                dict_ = get_res_doc_fac_pro_toc(context, vllm_endpoint, logfile)
                if dict_:
                    encounter_date = dict_["DOS"]
                else:
                    encounter_date = ""
                tocRangeList.append({"page_range": data['page_range'], "encounter_date": encounter_date})
        else:
            encounter_date = ""
            # start_dos -> 0:start vs, 1:end vs,2:no vs, 3:start & end vs
            for index, data in enumerate(tqdm(predictRange if predictRange else dos_range)):
                try:
                    splitRr = data["page_range"].split("-")
                    startRr = int(splitRr[0]) - 1
                    endRr = int(splitRr[1])
                    encounter_name = data["encounter_name"]
                    isStartDOS = data["start_dos"] if "start_dos" in data else 2

                    if predictRange:
                        whIndex = [data for dos in dos_list for data in all_text[startRr:endRr] if dos in data.lower()]
                        whIndex = list(filter(None, whIndex))
                        if not whIndex:
                            whIndex = [data for dos in dos_list1 for data in all_text[startRr:endRr] if
                                       dos in data.lower()]
                        whIndex = list(filter(None, whIndex))
                        context = "\n".join(whIndex)

                        indexEn = [{"start": m.start(0), "end": m.end(0)} for dos in dos_list for m in
                                   re.finditer(dos, context.lower())]
                        dates_se = []
                        indexEn = sorted(indexEn, key=lambda x: x["start"])
                        # CHECK IF TAKEN INDEX HAVE DATES
                        for date in indexEn:
                            dateCon = context[date["start"]:date["end"] + 150]
                            stop_words = list(set(stopwords.words('english')))
                            for sp in stop_words:
                                dateCon = re.sub(fr"\b{sp}\b", "", dateCon)
                            dates_se = search_dates(dateCon, settings={'STRICT_PARSING': True, 'DATE_ORDER': 'MDY'})
                            if dates_se:
                                break
                        # WHEN ENCOUNTER DATE OR ENC. DATE IS NOT AVAILABLE
                        if len(whIndex) == 0 or not dates_se:
                            whIndex = [data for dos in dos_list for data in all_text[startRr:endRr] if
                                       dos in data.lower()]
                        whIndex = list(filter(None, whIndex))
                        context = "\n".join(whIndex)

                    ### GET ENCOUNTER DATE
                    encounter_date = ""
                    if encounter_name:
                        # START & END VS
                        if isStartDOS == 3:
                            enc_list = []

                            for enc_ in encounter_name:
                                enc = toc_encounter_date1(all_text, startRr, endRr, enc_, vllm_endpoint, logfile)
                                enc_list.append(enc)
                            if len(enc_list) > 1:
                                if enc_list[0] == enc_list[1]:
                                    encounter_date = enc_list[0]
                                else:
                                    date1 = parser.parse(enc_list[0])
                                    date2 = parser.parse(enc_list[1])
                                    date_diff = (date2 - date1).days
                                    if 0 > date_diff > 20:
                                        encounter_date = enc_list[1]
                                    else:
                                        encounter_date = enc_list[0]
                            else:
                                encounter_date = enc_list[0] if enc_list else ""
                        else:
                            encounter_date = toc_encounter_date1(all_text, startRr, endRr, encounter_name,
                                                                 vllm_endpoint, logfile)
                    else:
                        date_list = []
                        context = "\n".join(all_text[startRr:endRr])
                        encounter_date = toc_encounter_date_vllm(context, "", f"{startRr + 1}-{endRr}", vllm_endpoint,
                                                                 logfile)
                        date_list.append(encounter_date)
                        date_list = list(set(date_list))
                        gp_date_list = []
                        for key, gp in groupby(date_list, key=lambda x: x):
                            gp_list = list(gp)
                            gp_date_list.append({"len": len(gp_list), "list": gp_list})
                        sorted(gp_date_list, key=lambda x: x["len"], reverse=True)
                        encounter_date = gp_date_list[0]["list"][0]

                    if encounter_date is None:
                        encounter_date = ""

                    ###CONTEXT APPEND
                    tocRangeList.append({"page_range": data['page_range'], "encounter_date": encounter_date})
                    # tocRangeList.append(
                    #     {"page_range": pageRange['page_range'], "encounter_date": encounter_date, "context": context})
                except Exception as e:
                    log_exception("toc_extraction predict dos range", logfile)

        if tocRangeList:
            try:
                # COMBINE DATE RANGE
                dos_range = []
                for key, gp in groupby(tocRangeList, key=lambda x: x["encounter_date"]):
                    gp_list = list(gp)
                    start = int((gp_list[0]["page_range"]).split('-')[0])
                    end = int((gp_list[len(gp_list) - 1]["page_range"]).split('-')[1])
                    dos_range.append({"page_range": f"{start}-{end}", "encounter_date": key})
                return dos_range
            except:
                return tocRangeList
        else:
            return tocRangeList
    except Exception as e:
        log_exception("toc_extraction", logfile)
        return []
