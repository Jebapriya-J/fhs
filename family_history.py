import json
import re
import sys
from datetime import datetime
from itertools import groupby

import requests
from dateparser.search import search_dates
from nltk.tokenize import word_tokenize
from rapidfuzz import fuzz
from tqdm import tqdm

headers = {
    "Content-Type": "application/json",
}

with open('config.json', 'r') as f:
    config_data = json.load(f)

vllm_endpoint = config_data["vllm_endpoint"]


def log_exception(e, func_name, logfile):
    exc_type, exc_obj, tb = sys.exc_info()
    lineno = tb.tb_lineno if tb else "Unknown"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = (
        f"\n[{timestamp}] "
        f"In {func_name} LINE.NO-{lineno} : {exc_obj} --> error {e}"
    )
    print("error_message :", error_message)
    with open(logfile, 'a', encoding='utf-8') as fp:
        fp.write(error_message)


def processLogger(process, logfile):
    with open(logfile, 'a', encoding='utf-8') as fp:
        fp.writelines(f'\n{datetime.now()} {process}')


def matchCordinatesFromWordCordinates(datas, word_coordinates, logfile):
    try:
        for data in datas:
            relation_match_found = False  # Track if a match is found for 'relation name'
            living_status_match_found = False  # Track if a match is found for 'relation name'
            cause_of_death_match_found = False  # Track if a match is found for 'relation name'
            comments_match_found = False  # Track if a match is found for 'relation name'
            reg = r'\b\w+\b'
            for word_coordinate in word_coordinates:
                if (
                        "family history" == data['report_type'].lower()
                        and data['acc_page_num'] == word_coordinate["Page"]
                ):

                    # Check for relation name match
                    if data['relation']:
                        relationIndex = re.findall(reg, data["relation"])
                        isrelationIndex = False
                        for md in relationIndex:
                            if md.strip().lower() in word_coordinate["text"].strip().lower():
                                isrelationIndex = True
                                break
                        if isrelationIndex:
                            # if data['relation name'].strip() and data['relation name'].strip().lower() == word_coordinate["text"].strip().lower():
                            data['relation_coordinates'] = {
                                'x0': word_coordinate["x0"],
                                'y0': word_coordinate["y0"],
                                'x1': word_coordinate["x1"],
                                'y1': word_coordinate["y1"],
                                'height': word_coordinate["height"],
                                'width': word_coordinate["width"]
                            }
                            relation_match_found = True  # Mark match found for 'relation name'
                            # break

                    # Check for living status name match
                    if data['living_status']:
                        living_statusIndex = re.findall(reg, data["living_status"])
                        isliving_statusIndex = False
                        for md in living_statusIndex:
                            if md.strip().lower() in word_coordinate["text"].strip().lower():
                                isliving_statusIndex = True
                                break
                        if isliving_statusIndex:
                            data['living_status_coordinates'] = {
                                'x0': word_coordinate["x0"],
                                'y0': word_coordinate["y0"],
                                'x1': word_coordinate["x1"],
                                'y1': word_coordinate["y1"],
                                'height': word_coordinate["height"],
                                'width': word_coordinate["width"]
                            }
                            living_status_match_found = True  # Mark match found for 'relation name'
                            # break

                    # Check for cause of death name match
                    if data['cause_of_death']:
                        cause_of_deathIndex = re.findall(reg, data["cause_of_death"])
                        iscause_of_deathIndex = False
                        for md in cause_of_deathIndex:
                            if md.strip().lower() in word_coordinate["text"].strip().lower():
                                iscause_of_deathIndex = True
                                break
                        if iscause_of_deathIndex:
                            # if data['relation name'].strip() and data['relation name'].strip().lower() == word_coordinate["text"].strip().lower():
                            data['cause_of_death_coordinates'] = {
                                'x0': word_coordinate["x0"],
                                'y0': word_coordinate["y0"],
                                'x1': word_coordinate["x1"],
                                'y1': word_coordinate["y1"],
                                'height': word_coordinate["height"],
                                'width': word_coordinate["width"]
                            }
                            cause_of_death_match_found = True  # Mark match found for 'relation name'
                            # break

                    # Check for comments name match
                    if data['comments']:
                        commentsIndex = re.findall(reg, data["comments"])
                        iscommentsIndex = False
                        for md in commentsIndex:
                            if md.strip().lower() in word_coordinate["text"].strip().lower():
                                iscommentsIndex = True
                                break
                        if iscommentsIndex:
                            # if data['relation name'].strip() and data['relation name'].strip().lower() == word_coordinate["text"].strip().lower():
                            data['comments_coordinates'] = {
                                'x0': word_coordinate["x0"],
                                'y0': word_coordinate["y0"],
                                'x1': word_coordinate["x1"],
                                'y1': word_coordinate["y1"],
                                'height': word_coordinate["height"],
                                'width': word_coordinate["width"]
                            }
                            comments_match_found = True  # Mark match found for 'relation name'
                            # break

            # If no match was found for 'relation name', add default coordinates
            if not relation_match_found:
                data['relation_coordinates'] = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0, 'height': 0, 'width': 0}

            if not living_status_match_found:
                data['living_status_coordinates'] = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0, 'height': 0, 'width': 0}

            if not cause_of_death_match_found:
                data['cause_of_death_coordinates'] = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0, 'height': 0, 'width': 0}

            if not comments_match_found:
                data['comments_coordinates'] = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0, 'height': 0, 'width': 0}
        return datas
    except Exception as e:
        log_exception(e,"In tensore_llama_3_11 matchCordinatesFromWordCordinates", logfile)


def check_match_words(word, page_num, word_coordinates, isFuzz, logfile):
    try:
        reg = r'[^A-Za-z0-9]'
        word_list = []
        for i, coordinate in enumerate(word_coordinates):
            if not isinstance(coordinate, dict):
                continue
            try:
                if isFuzz:
                    re_word = re.sub(reg, "", word.lower())
                    re_text = re.sub(reg, "", coordinate["text"].lower())
                    isTrue = fuzz.partial_ratio(re_text, re_word) > 90

                else:
                    isTrue = re.sub(reg, "", coordinate["text"].lower()) == re.sub(reg, "", word.lower())

                if isTrue:
                    if bool(re.search(re.sub(reg, "", word.lower()), re.sub(reg, "", coordinate["text"].lower()))):
                        word_list.append({
                            'x0': coordinate["x0"],
                            'y0': coordinate["y0"],
                            'x1': coordinate["x1"],
                            'y1': coordinate["y1"],
                            'height': coordinate["height"],
                            'width': coordinate["width"],
                            'page_num': coordinate["Page"],
                        })
            except Exception as e:
                print(f"[DEBUG] index {i}: type={type(coordinate)}, value={coordinate} , error={e}")
                log_exception(e, "check_match_words", logfile)

        return word_list
    except Exception as e:
        log_exception(e, "check_match_words", logfile)
        return []


def check_match_coordinates(word_list, check_word_coord, page_num, logfile):
    try:
        data = {}
        if word_list:
            # REMOVE DUP
            seen = set()
            non_dup = []
            for non_data in word_list:
                key = (non_data["x0"], non_data['y0'], non_data["x1"], non_data['y1'], non_data['height'],
                       non_data['width'], non_data['page_num'])
                if not key in seen:
                    seen.add(key)
                    non_dup.append(non_data)
                    # print(non_dup)
            if check_word_coord:
                if not check_word_coord["y0"] == 0:
                    check_coord = []
                    for coord in non_dup:
                        if check_word_coord["y0"] < coord["y0"]:
                            check_coord.append(coord)
                            break
                    if check_coord:
                        data = {
                            'x0': check_coord[0]["x0"],
                            'x1': check_coord[0]["x1"],
                            'y0': check_coord[0]["y0"],
                            'y1': check_coord[0]["y1"],
                            'height': check_coord[0]["height"],
                            'width': check_coord[0]["width"],
                            'page_num': check_coord[0]["page_num"]
                        }
                    else:
                        data = {
                            'x0': 0,
                            'x1': 0,
                            'y0': 0,
                            'y1': 0,
                            'height': 0,
                            'width': 0,
                            'page_num': page_num,
                        }
                else:
                    data = {
                        'x0': non_dup[0]["x0"],
                        'x1': non_dup[0]["x1"],
                        'y0': non_dup[0]["y0"],
                        'y1': non_dup[0]["y1"],
                        'height': non_dup[0]["height"],
                        'width': non_dup[0]["width"],
                        'page_num': non_dup[0]["page_num"]
                    }
            else:

                data = {
                    'x0': non_dup[0]["x0"],
                    'x1': non_dup[0]["x1"],
                    'y0': non_dup[0]["y0"],
                    'y1': non_dup[0]["y1"],
                    'height': non_dup[0]["height"],
                    'width': non_dup[0]["width"],
                    'page_num': non_dup[0]["page_num"]
                }
        else:
            data = {
                'x0': 0,
                'x1': 0,
                'y0': 0,
                'y1': 0,
                'height': 0,
                'width': 0,
                'page_num': page_num,
            }

        return data

    except Exception as e:
        log_exception(e, "check_match_coordinates", logfile)
        return {}


def match_coordinates(output, word_coordinates, logfile):
    try:
        new_data = []
        reg = r'[^A-Za-z0-9]'
        for data in output:
            if data["report_type"] == "family history":
                page_num = data["acc_page_num"]
                # print("WORDS_COORDINATES LENGTH:", len(word_coordinates))
                # print("TRYING TO ACCESS PAGE:", page_num)
                # RELATION
                word_list = check_match_words(data["relation"], page_num, word_coordinates[page_num - 1], True, logfile)
                fin_coord = check_match_coordinates(word_list, {}, page_num, logfile)
                data["relation_coordinates"] = fin_coord

                # LIVING STATUS
                word_list = check_match_words(data["living_status"], page_num, word_coordinates[page_num - 1], True,
                                              logfile)
                fin_coord = check_match_coordinates(word_list, {}, page_num, logfile)
                data["living_status_coordinates"] = fin_coord

                # COMMENTS
                word_list = check_match_words(data["comments"], page_num, word_coordinates[page_num - 1], True, logfile)
                fin_coord = check_match_coordinates(word_list, {}, page_num, logfile)
                data["comments_coordinates"] = fin_coord

                # CASE OF DEATH
                word_list = check_match_words(data["cause_of_death"], page_num, word_coordinates[page_num - 1], True,
                                              logfile)
                fin_coord = check_match_coordinates(word_list, {}, page_num, logfile)
                data["cause_of_death_coordinates"] = fin_coord
                new_data.append(data)
        return new_data
    except Exception as e:
        log_exception(e, "match_coordinates", logfile)
        return output


def extract_json(all_text, response, logfile):
    try:
        reg_pattern = r'\[\s*{.*?}\s*\]'
        all_data = []
        # relationList = ["sibling", "siblings", "kid", "kids", "son", "sons", "daughter", "daughters", "sister",
        #                 "sisters", "brother", "brothers", "father", "mother", "dad", "mom","dads","moms", "wife", "husband",
        #                 "grandfather", "grandmother", "paternalgrandfather", "paternalgrandmother","bro","sis",
        #                 "maternalgrandfather", "maternalgrandmother", "grandfatherpaternal", "grandmotherpaternal",
        #                 "grandfathermaternal", "grandmothermaternal","grandparent", "grandparents"]
        relationList = ["sibling", "siblings", "sister",
                        "sisters", "brother", "brothers", "father", "bro", "sis", "mother", "dad", "mom", "dads",
                        "moms",
                        "grandfather", "grandmother", "paternalgrandfather", "paternalgrandmother",
                        "maternalgrandfather", "maternalgrandmother", "grandfatherpaternal", "grandmotherpaternal",
                        "grandfathermaternal", "grandmothermaternal", "grandparent", "grandparents"]
        relationDict_list = [["dad", "dads", "father", "fathers"],
                             ["mom", "moms", "mother", "mothers"],
                             ["sister", "sisters", "brother", "brothers", "bro", "sis"],
                             ["grandmother", "grandfather", "paternalgrandfather",
                              "paternalgrandmother",
                              "maternalgrandfather", "maternalgrandmother", "grandfatherpaternal",
                              "grandmotherpaternal",
                              "grandfathermaternal", "grandmothermaternal", "grandparent",
                              "grandparents",
                              "maternal grand father", "maternal grand mother",
                              "paternal grand father",
                              "paternal grand mother"]]
        regBn = r"[^A-Za-z0-9]"
        for result in response:
            text = result["output"]
            # print(text)
            page = result["Page"]
            # print(page)
            textRe = re.findall(reg_pattern, text, re.DOTALL)
            # print("1111")
            if textRe:
                replaceText = textRe[0]
                reText = replaceText.lower().replace("empty string", "").replace("empty string", "")
                # print("1111345")
                try:
                    jsonData = json.loads(reText)
                    # print("111111")
                    for data in jsonData:
                        try:
                            page = result["Page"]
                            encounter_date = result["encounter_date"]
                            relation = data["relation"]
                            relationSp = re.sub(r"\s+", " ", re.sub(regBn, " ", relation))
                            context = all_text[page - 1].lower()
                            context = context.replace("\n", " ")
                            context = re.sub(r"\s+", " ", re.sub(regBn, " ", context))
                            isFamily = False
                            for rel in relationList:
                                if bool(re.search(fr"\b{rel.lower()}\b", context)):
                                    isFamily = True
                                    break

                            if relationSp and data["report_type"] and isFamily and [tag for tag in
                                                                                    relationSp.lower().split() if
                                                                                    tag in relationList]:
                                # if relation and data["report_type"]:
                                status = data["living_status"].strip()
                                if "dead" in status or "decease" in status:
                                    status = "dead"
                                cause = data["cause_of_death"].strip() if "cause_of_death" in data else ""
                                comments = data["comments"].strip()
                                age = str(data["age"]).strip() if "age" in data else ""
                                count = str(data["count"]).strip() if "count" in data else ""
                                relation = data["relation"].strip()
                                # print("relation",relation)
                                # CHANGE RELATIONS
                                isRelation = False

                                # GRAND PARENTS
                                if fuzz.partial_ratio("grand", relation.lower()) > 90:
                                    isRelation = True
                                    relation = "Grand Parents"
                                # FATHER
                                elif [rel for rel in relation.split() if rel in relationDict_list[0]]:
                                    relation = "Father"
                                # MOTHER
                                elif [rel for rel in relation.split() if rel in relationDict_list[1]]:
                                    relation = "Mother"
                                # SIBLINGS
                                elif [rel for rel in relation.split() if rel in relationDict_list[2]]:
                                    isRelation = True
                                    relation = "Siblings"

                                # CHECK LIVING STATUS
                                if status.lower() == "n":
                                    status = "alive"
                                elif status.lower() == "y":
                                    status = "dead"

                                # CHECK COMMENTS AND CAUSE OF DEATH
                                if status:
                                    if "live" in status and cause:
                                        data["cause_of_death"] = ""
                                        # ADD RELATION IN COMMENTS
                                        if isRelation:
                                            if cause:
                                                cause = data["relation"] + " has " + cause
                                            else:
                                                cause = data["relation"]
                                        data["comments"] = cause
                                    elif "dead" in status and not cause:
                                        # ADD RELATION IN COMMENTS
                                        if isRelation:
                                            if comments:
                                                comments = data["relation"] + " has " + comments
                                            else:
                                                comments = data["relation"]

                                        data["cause_of_death"] = comments
                                        data["comments"] = ""
                                    elif "dead" in status and cause:
                                        # ADD RELATION IN COMMENTS
                                        if isRelation:
                                            if cause:
                                                cause = data["relation"] + " has " + cause
                                            else:
                                                cause = data["relation"]

                                        data["cause_of_death"] = cause
                                    else:
                                        # ADD RELATION IN COMMENTS
                                        if isRelation:
                                            if comments:
                                                comments = data["relation"] + " has " + comments
                                            else:
                                                comments = data["relation"]
                                        data["comments"] = comments
                                else:
                                    if cause:
                                        data["cause_of_death"] = ""
                                        # ADD RELATION IN COMMENTS
                                        if isRelation:
                                            if cause:
                                                cause = data["relation"] + " has " + cause
                                            else:
                                                cause = data["relation"]
                                        data["comments"] = cause
                                    else:
                                        # ADD RELATION IN COMMENTS
                                        if isRelation:
                                            if comments:
                                                comments = data["relation"] + " has " + comments
                                            else:
                                                comments = data["relation"]
                                        data["comments"] = comments

                                # REMOVE ALPHA IN AGE AND COUNT
                                regNum = r"[^0-9]"
                                count = re.sub(r"\s+", " ", re.sub(regNum, " ", count))
                                # CHECK AGE HAS MONTHS WEEK DAYS
                                if [ag for ag in re.sub(regBn, " ", age) if ag in ["day,days,week,weeks,month,months"]]:
                                    age = 0
                                else:
                                    age = re.sub(r"\s+", " ", re.sub(regNum, " ", age))
                                data["living_status"] = status
                                data["relation"] = relation
                                data["acc_page_num"] = page
                                data["count"] = "" if str(count) == "0" else count
                                data["age"] = "" if str(age) == "0" else age
                                if encounter_date:
                                    data["date"] = encounter_date
                                else:
                                    date = data["date"]
                                    tDate = search_dates(date, settings={'STRICT_PARSING': True})
                                    if tDate:
                                        if tDate[0][1].year > 2000:
                                            date = tDate[0][1].strftime("%Y-%m-%d")
                                    else:
                                        date = ""
                                    data["date"] = date
                                data["report_type"] = "family history"
                                all_data.append(data)
                                # print("all", all_data)
                        except Exception as e:
                            log_exception(e, f"for loop error:{data}", logfile)

                except Exception as e:
                    log_exception(e, f"extract json loads error:{page}", logfile)
        all_data = sorted(all_data, key=lambda x: x["acc_page_num"])
        # print("all_data",all_data)
        # APPEND RELATIONS FATHER AND MOTHER COMMENTS AND CAUSE
        gp_data = groupby(all_data, key=lambda x: x["acc_page_num"])
        fin_data = []
        rel_list = ["mother", "father", "mothers", "fathers", "mom", "dad", "moms", "dads"]
        for key, gp in gp_data:
            ls = list(gp)
            ls = sorted(ls, key=lambda x: x["relation"])
            for relkey, relgp in groupby(ls, key=lambda x: x["relation"]):
                rells = list(relgp)
                if relkey in rel_list:
                    if len(rells) > 1:
                        jsData = {}
                        age = ""
                        status = ""
                        cause = ""
                        comments = ""
                        for i, data in enumerate(rells):
                            age = data["age"] if age else ""
                            status = data["living_status"] if status else ""
                            if i == len(rells) - 1:
                                cause += f' {data["cause_of_death"]} '
                                comments += f' {data["comments"]} '
                            else:
                                cause += f' {data["cause_of_death"]} ,' if data["cause_of_death"] else ""
                                comments += f' {data["comments"]} ,' if data["comments"] else ""
                        cause = cause.strip()
                        comments = comments.strip()

                        if cause and cause[len(cause) - 1] == ",":
                            cause = cause.removesuffix(",")

                        if comments and comments[len(comments) - 1] == ",":
                            comments = comments.removesuffix(",")

                        jsData["relation"] = rells[0]["relation"]
                        jsData["age"] = age
                        jsData["living_status"] = status
                        jsData["cause_of_death"] = cause.strip()
                        jsData["comments"] = comments.strip()
                        jsData["report_type"] = rells[0]["report_type"]
                        jsData["count"] = 1
                        jsData["date"] = rells[0]["date"]
                        jsData["acc_page_num"] = key

                        fin_data.append(jsData)
                    else:
                        fin_data.extend(rells)
                else:
                    fin_data.extend(rells)
        return fin_data
    except Exception as e:
        log_exception(e, "extract_json", logfile)

# def vllmAPI_local(context,prompt,logfile):
#     try:
       
#         url = "http://172.16.36.10:9002/"
 
#         payload = json.dumps({
#         "prompt": prompt,
#         "text": context
#         })
#         headers = {
#         'Content-Type': 'application/json'
#         }
 
#         response = requests.request("POST", url, headers=headers, data=payload)
 
#         res= response.text
#         res=res.replace("\\n","").replace("\\","")
#         return res
#     except Exception as e:
#         log_exception("vllmapi1:",logfile
     
def vllmAPI(content, context, task, logfile,vllm_endpoint, max_token):
    try:
        url=vllm_endpoint
        input_prompt = f"Context : {context} \n\n task : {task}\n Answer:"

        payload = {
            "messages": [
                {"role": "system",
                 "content": content},
                {"role": "user", "content": input_prompt},
            ],
            "stream": False,
            "max_tokens": max_token,
            "stop": None,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "temperature": 0,
            "top_p": 0.95
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        result = response.json()
        if str(response.status_code) != "200":
            processLogger(f"{result}", logfile)
        
        content = result['choices'][0]['message']['content']
        # print(content)
        return content
    except Exception as e:
        log_exception(e, "vllmAPI", logfile)


def family_history_extraction(all_text, tocRangeList, words_coordinates,vllm_endpoint, logfile):
    try:

        task = """
                Task:
                    Extract all family history related data from the provided text and structure it in JSON format following these strict guidelines:
                Scope of Extraction:
                    Only rely on the information explicitly present in the given text.
                    Strictly Avoid outside information from the context.
                    Strictly Avoid hallucination data from outside context.
                Extraction Rules:
                    The main important rule is to extract the fields only when there is FAMILY HISTORY section or title.
                    Compulsory use all fields in below instructions,do not skip fields in context.
                    Ensure that your reply follows the following valid JSON format with NO PREAMBLE and POSTAMBLE:
                    For each family history entry, capture the following fields:
                    ```[{
                            "relation": "",
                            "count": "",
                            "age": "",
                            "living_status": "",
                            "cause_of_death": "",
                            "comments": "",
                            "report_type": "",
                            "date":""
                    }]```

                Detailed Field Descriptions:
                    "relation": "relationship between patient like neg hx, husband,wife,son,father,mother,sister,brother,siblings,children,maternal grand father,maternal grand mother,paternal grand father,paternal grand mother , return valid relation for patient or else return empty string".
                    "count": "total counts for relationship brother,sister,kids,siblings,children,son,daughter return only integer value or else return empty string".
                    "age": "relations age return only integer value or else return empty string".
                    "cause_of_death": "cause of death for relations or else return empty string".
                    "comments": "problems or disease or diabetics for relations given comments or conditions and add problem or else return empty string".
                    "living_status": "status for relation alive or deceased or "", return alive or deceased or "", if status have in context or else return empty string".
                    "report_type": "above fields is available means return family history or else return empty string".
                    "date":"Dates may appear under labels such as `"Visit Date"`, `"Visit"`, `"Visited on"`. **Prioritize the visit date first** and normalize to the format `YYYY-MM-DD`."
                Exclusion Criteria:
                    Ensure all the above fields are mapped correctly.
                    If the text does not contain family history related data, return an empty JSON array ([]).
                    Do not include symptoms, medical conditions, social history,or unrelated doctor notes.
                """

        content = "You are an expert in extracting all family history reports,your task is to get the data from the provided context as per the given task."
        all_data = []
        if tocRangeList:
            for pageRange in tqdm(tocRangeList):
                splitRr = pageRange["page_range"].split("-")
                startRr = int(splitRr[0]) - 1
                endRr = int(splitRr[1])
                encounter_date = pageRange["encounter_date"]
                for page_num in range(startRr, endRr):
                    context = all_text[page_num]
                    tokenLength = len(word_tokenize(context))
                    if tokenLength > 100000:
                        context_length = 50000
                    else:
                        context_length = 8192
                    response = vllmAPI(content, context, task, logfile,vllm_endpoint, context_length)
                    all_data.append({"output": response, "Page": page_num + 1, "encounter_date": encounter_date})
                    print("55555555555555555",all_data)
        else:
            for page_num, context in enumerate(tqdm(all_text)):
                tokenLength = len(word_tokenize(context))
                if tokenLength > 100000:
                    context_length = 50000
                else:
        
                    context_length = 8192
                response = vllmAPI(content, context, task, logfile,vllm_endpoint, context_length)

                all_data.append({"output": response, "Page": page_num + 1, "encounter_date": ""})
                
        ##EXTRACT JSON
        output = extract_json(all_text, all_data, logfile)
        print("2222222222",output)
        if output:
            # output = matchCordinatesFromWordCordinates(output, words_coordinates, logfile)
            output = match_coordinates(output, words_coordinates, logfile)
            processLogger(f"Family History len:{len(output)}", logfile)

            print(f"family len:{len(output)}")
            # print(output)
        else:
            print("Family History is Empty")
            processLogger("Family History is Empty", logfile)
        return output
    except Exception as e:
        log_exception(e, "family_history_extraction", logfile)
import nltk
# nltk.download('punkt_tab')
import os
# import pdfplumber
# from datetime import timedelta
from toc_range import toc_extraction


def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    for key in ["INPUT_FOLDER","OUTPUT_FOLDER", "LOG_FOLDER"]:
        folder = config[key]
        if not os.path.exists(folder):
            os.makedirs(folder)
            # print(f"Created folder: {folder}")
        # else:
        #     # print(f"Folder already exists: {folder}")
    return config

config=load_config()
UPLOAD_FOLDER = config.get("INPUT_FOLDER", "input")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


if __name__ == "__main__":
    filename="18846"
    # file_path=r"/home/aravinth.chinnasamy/jebapriya/input/18308.json"
    js_path=rf"/home/aravinth.chinnasamy/jebapriya/input/{filename}.json"
    all_res = json.loads(open(js_path,"r+").read())
    all_texts,all_word_coordinates = all_res["texts"],all_res["coordinates"]
    print("text", all_texts)
    tocRangeList = toc_extraction(
            all_text=all_texts,
            file=filename,
            vllm_endpoint=vllm_endpoint,
            logfile=config["LOG_FILE"]
        )
    # tocRangeList=[]
    family = family_history_extraction(
        all_text=all_texts,
        tocRangeList=tocRangeList,
        words_coordinates=[],
        vllm_endpoint=vllm_endpoint,
        logfile=config["LOG_FILE"]
    )

    print("\n====== FAMILY HISTORY OUTPUT ======\n")
    print(json.dumps(family, indent=2))