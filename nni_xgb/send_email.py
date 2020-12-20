import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.base import MIMEBase
from email import encoders
import logging
from datetime import datetime,timedelta
import glob
import os

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
logger = logging.getLogger(os.path.basename(__file__))

smtpserver = '*.*.*.*'
smtpport = '*'

def get_best_model():
    yesterday = datetime.today()+timedelta(-1)
    yesterday_format = yesterday.strftime('%Y-%m-%d')
    try:
       best_model = glob.glob('{}/*.model'.format(yesterday_format))[0]
       best_model = best_model.split('/')[-1].split('_')[-1]
    except:
       best_model = 'no model'
       logger.info('no model')
    return best_model
     
def send_event_email(mail_title,sender,receiver,ccer,best_model):

    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = receiver
    message["Cc"] = ''
    message["Subject"] = Header(mail_title, "utf-8")
    receiver_list = receiver.split(",")
    ccer_list = ccer.split(",")
    toaddrs = receiver_list + ccer_list
    body =""
    body+= 'best automl xgb model:\n{}'.format(best_model)
    txt = MIMEText(body, 'html', 'utf-8')
    message.attach(txt)
    try:
        smtpobj = smtplib.SMTP(smtpserver, port=smtpport)
        smtpobj.sendmail(sender, toaddrs, message.as_string())
        logger.info("邮件发送成功.")
        smtpobj.quit()
    except smtplib.SMTPException as ex:
        logger.error(ex)

if __name__=='__main__':

  title = 'automl-xgb'
  sender = '*@**.com.cn'
  receiver = '*@*.com.cn'
  ccer = '*@*.com.cn'
  best_model = get_best_model()
  send_event_email(title, sender, receiver, ccer,best_model)

