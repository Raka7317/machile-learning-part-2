#!/usr/bin/env python
# coding: utf-8

# In[28]:


import requests


# In[29]:


from datetime import datetime, timedelta


# In[30]:


import time


# In[31]:


import pytz


# In[32]:


from os import environ


# In[33]:


Define all the constants
time_interval=20
PINCODE="206201"
msg="blank"
tele_auth_token=
tel_gruop_id=
IST=pytz.timezone('Asia/Kolkata')
slot_found=False
header={'User-Agent': 'Chrome/84.0.4147.105 safari/537.36'}


# In[34]:


def update_timestamp_send_Request(PINCODE):
     raw_TS = datetime.now(IST) + timedelta(days=1)      # Tomorrows date
     tomorrow_date = raw_TS.strftime("%d-%m-%Y")         # Formatted Tomorrow's date
     today_date = datetime.now(IST).strftime("%d-%m-%Y") #Current Date
     curr_time = (datetime.now().strftime("%H:%M:%S"))   #Current time
     request_link = f"https://cdn-api.co-vin.in/api/v2/appointment/sessions/public/calendarByPin?pincode={PINCODE}&date={tomorrow_date}"
     response = requests.get(request_link, headers = header)
     raw_JSON = response.json()
     return raw_JSON, today_date, curr_time


# In[35]:


def get_availability_45(age = 45):
    raw_JSON, today_date, curr_time = update_timestamp_send_Request(PINCODE)
    for cent in raw_JSON['centers']:
        for sess in cent["sessions"]:
            sess_date = sess['date']
            if sess["min_age_limit"] == age and sess["available_capacity"] > 0:
                slot_found =  True
                msg = f"""For age 45+ [Vaccine Available] at {PINCODE} on {sess_date}\n\tCenter : {cent["name"]}\n\tVaccine: {sess["vaccine"]}\n\tDose_1: {sess["available_capacity_dose1"]}\n\tDose_2: {sess["available_capacity_dose2"]}"""
                send_msg_on_telegram(msg)
                print (f"INFO:[{curr_time}] Vaccine Found for 45+ at {PINCODE}")
    else:
        slot_found =  False
        print (f"INFO: [{today_date}-{curr_time}] Vaccine NOT-Found for 45+ at {PINCODE}")


# In[36]:


def get_availability_18(age = 18):
     raw_JSON, today_date, curr_time = update_timestamp_send_Request(PINCODE)
     for cent in raw_JSON['centers']:
         for sess in cent["sessions"]:
             sess_date = sess['date']
             if sess["min_age_limit"] == age and sess["available_capacity"] > 0:
                 slot_found =  True
                 msg = f"""For age 18+ [Vaccine Available] at {PINCODE} on {sess_date}\n\tCenter : {cent["name"]}\n\tVaccine: {sess["vaccine"]}\n\tDose_1: {sess["available_capacity_dose1"]}\n\tDose_2: {sess["available_capacity_dose2"]}"""
                 send_msg_on_telegram(msg)
                 print (f"INFO: [{curr_time}] Vaccine Found for 18+ at {PINCODE}")
     else:
         slot_found =  False
         print (f"INFO: [{today_date}-{curr_time}] Vaccine NOT Found for 18+ at {PINCODE}")


# In[37]:


def send_msg_on_telegram(msg):
    telegram_api_url = f"https://api.telegram.org/bot{tele_auth_token}/sendMessage?chat_id=@{tel_group_id}&text={msg}"
    tel_resp = requests.get(telegram_api_url)
if tel_resp.status_code == 200:
    print ("Notification has been sent on Telegram") 
else:
    print ("Could not send Message")


# In[38]:


if name == "main":    
    while True:
        get_availability_45()
        get_availability_18()
        time.sleep(time_interval)


# In[ ]:





# In[ ]:




