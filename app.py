# -*- coding: utf-8 -*-

#  Licensed under the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License. You may obtain
#  a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations
#  under the License.

from __future__ import unicode_literals

import errno

import os
import glob
from dotenv import load_dotenv

load_dotenv()
import sys
import tempfile
from argparse import ArgumentParser

from flask import Flask, request, abort

import tensorflow as tf
from tensorflow import keras
import numpy as np

from PIL import Image

model = tf.keras.models.load_model("model_Pre-train_MobileNetV2", compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Declare a variable
CLASS_NAMES = [
    "โรคแอนแทรคโนส",
    "โรคแคงเกอร์",
    "โรคใบแห้ง",
    "โรคบั่วปมใบมะม่วง",
    "ไม่เป็นโรค",
    "โรคราแป้ง",
    "โรคราดำมะม่วง",
]

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextSendMessage,
    ImageMessage,
)
import serverless_wsgi

app = Flask(__name__)

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv("LINE_CHANNEL_SECRET", None)
channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", None)
if channel_secret is None:
    print("Specify LINE_CHANNEL_SECRET as environment variable.")
    sys.exit(1)
if channel_access_token is None:
    print("Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.")
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

static_tmp_path = os.getenv("STATIC_TMP_PATH", os.path.join(os.path.dirname(__file__), "tmp"))

def lambda_handler(event, context):
    return serverless_wsgi.handle_request(app, event, context)


# function for create tmp dir for download content
def make_static_tmp_dir():
    try:
        os.makedirs(static_tmp_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
            pass
        else:
            raise


def delete_pic():
    removing_files = glob.glob("./tmp/*.jpg")
    for i in removing_files:
        os.remove(i)


def about_pic(image_path):
    img = Image.open(open(image_path, "rb"))
    img = img.resize((180, 180))

    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    classes = model.predict(img)
    print("classes: " + str(classes))
    prediction = np.argmax(classes)
    print("prediction: " + str(prediction))

    ans = ""
    if CLASS_NAMES[prediction] == "โรคแอนแทรคโนส":
        ans = "โรคแอนแทรคโนส\nสาเหตุและอาการ\nสาเหตุเกิดจากเชื้อรา Colletotrichum gloeosporioides\nอาการของโรคเกิดได้กับส่วนต่างๆของพืชได้แก่ อาการใบจุด ใบและกิ่งแห้งตาย อาการทั้งหมดที่เกิดขึ้นนี้จะเป็นแหล่งสะสมเชื้อที่จะแพร่ระบาดเข้าทำลายผลมะม่วงที่กำลังพัฒนา โดยเชื้อจะอยู่ในระยะพักตัว จนกระทั่งมะม่วงแก่และถึงระยะเก็บเกี่ยว ต่อมาเมื่อผลใกล้สุก เชื้อจะเริ่มพัฒนาใหม่และทำลายผลมะม่วงก่อให้เกิดอาการผลเน่าเสีย \nวิธีรักษา \nประกอบด้วยการจัดการภายในสวน และการใช้สารเคมีในการป้องกันกำจัดโรคพืช\n1. การจัดการภายในสวน\nต้องมีการตัดแต่งกิ่งและทรงพุ่มให้โปร่ง เพราะต้นมะม่วงที่แตกกิ่งก้านหนาแน่น เป็นที่สะสมของโรคและแมลง เป็นอย่างดี แสงแดดส่องถึงทำให้บรรยากาศในพุ่มแห้ง ไม่เหมาะต่อการเจริญเติบโตของเชื้อโรค \nการตัดแต่งทรงพุ่มให้โปร่งมีส่วนเกี่ยวข้องกับสรีรวิทยาของต้น ช่วยให้ต้นมะม่วงจัดระบบการใช้แร่ธาตุอาหารได้ดี มีผลต่อการพัฒนาผลมะม่วง และยังเป็นวิธีที่ลดปริมาณการใช้สารเคมีได้\n2.การใช้สารเคมีในการป้องกันกำจัดโรคพืชสารเคมีที่มีประสิทธิภาพในการควบคุมโรคแอนแทรคโนสของมะม่วงประกอบด้วย 2 กลุ่มหลัก คือสารเคมีประเภทสัมผัส ได้แก่ แมนโคเซ็บ กลุ่มสารประกอบทองแดงหรือ คอปเปอร์ไฮดรอกไซด์ คอปเปอร์ซัลเฟต และ คอปเปอร์อ๊อกซี่คลอไรด์ สารเคมีประเภทดูดซึม ได้แก่ คาร์เบนดาซิม อะซ็อกซี่สโตรบิน และ โปรคลอราซ ทั้งนี้ไม่ควรใช้สารเคมีชนิดเดียวอย่างต่อเนื่องหรือติดต่อกันเป็นเวลานาน รวมทั้งไม่ใช้ในอัตราที่สูงเกินกว่ากำหนด เพราะจะมีผลต่อการพัฒนาความต้านทานต่อสารเคมีของเชื้อก่อโรค"
    elif CLASS_NAMES[prediction] == "โรคแคงเกอร์":
        ans = "โรคแคงเกอร์\nสาเหตุและอาการ\nแบคทีเรียที่ทำให้เกิดโรคแคงเกอร์ คือ Pseudomonas syringae จะเข้าสู่ต้นไม้ผ่านทางเปลือกไม้ที่ได้รับบาดเจ็บหรือบาดแผลที่มีอยู่ เช่น การตัดแต่งกิ่งไม้หรือกิ่งไม้ และสามารถแพร่กระจายโดยฝนหรือน้ำ และ เครื่องมือตัดแต่งกิ่ง\nวิธีรักษา\n1.ตัดแต่งกิ่งตอนดอกบานเมื่อแผลหายเร็ว\n2.นำกิ่งที่เหี่ยวหรือตายออกด้านล่างบริเวณที่ติดเชื้อ\n3.จัดการตัดแต่งกิ่งทั้งหมดทันทีและตรวจสอบให้แน่ใจว่าได้ฆ่าเชื้ออุปกรณ์ตัดแต่งกิ่งของคุณ - น้ำยาฟอกขาว 1 ส่วนต่อน้ำ 4 ส่วน - หลังการตัดแต่ละครั้ง\n4.หากใช้เครื่องเล็มหญ้ารอบโคนต้นไม้ หลีกเลี่ยงการทำลายเปลือกไม้เพื่อป้องกันการติดเชื้อ\n5.แปรงเปลือกไม้ด้วยน้ำยางสีขาวเจือจางด้วยน้ำเพื่อลดความผันผวนของอุณหภูมิที่ทำลายเปลือกไม้\n6.กำจัดวัชพืชและหญ้าออกจากรอบ ๆ โคนต้นไม้เล็กเพื่อให้อากาศไหลเวียนได้ดีขึ้นและทำให้ลำต้นแห้ง"
    elif CLASS_NAMES[prediction] == "โรคใบแห้ง":
        ans = "โรคใบแห้ง\nสาเหตุและอาการ\nโรคใบแห้งเป็นโรคเชื้อราที่ส่งผลต่อต้นมะม่วง ทำให้เกิดการติดเชื้อที่ใบและผล รวมถึงโรคปากนกกระจอกที่ลำต้นและกิ่ง โรคนี้ทำให้ผลผลิตและคุณภาพของมะม่วงลดลงอย่างมาก\nวิธีรักษา\nเทคนิคการควบคุมไอออน ซึ่งทำให้สภาพแวดล้อมบนใบพืชไม่เหมาะสมสำหรับการเจริญเติบโตของเชื้อรา สามารถทำได้โดยการผสม ไอเอส และ FK-1 แล้วฉีดพ่นพร้อมกันบนต้นไม้ที่ได้รับผลกระทบ ไอเอส เป็นสารประกอบอินทรีย์ที่ช่วยกำจัดโรค ในขณะที่ FK-1 เป็นสารควบคุมการเจริญเติบโตของพืชที่ช่วยเร่งการงอกใหม่ของพืชจากการทำลายของโรค นอกจากนี้ยังช่วยในการบำรุงและส่งเสริมการเจริญเติบโตและผลผลิตที่ดีขึ้น"
    elif CLASS_NAMES[prediction] == "โรคบั่วปมใบมะม่วง":
        ans = "โรคบั่วปมใบมะม่วง\nสาเหตุและอาการ\nเกิดจากแมลงที่เป็นศัตรูเข้าทำลายมีหลายชนิดเช่นเพลี้ยไก่ฟ้ามะม่วง ,แตนสร้างปม,แมลงบั่วมะม่วง\nลักษณะอาการใบมะม่วงจะเกิดเป็นปุ่มปมซึ่งแมลงชนิดดังกล่าวสร้างขึ้น มีลักษณะกลมๆ คล้ายกับเม็ดสาคูขนาดเส้นผ่าศูนย์กลาง 2 – 4 มิลลิเมตร เกิดจากแมลงวางไข่บนใบ ทำให้เนื้อเยื่อที่อยู่บริเวณรอบไข่จะขยายตัวออกนูนขึ้นบนแผ่นใบมองเห็นได้ชันเจนมีขนาดทั้งเล็กและใหญ่ปะปนกัน\nวิธีรักษา\nการป้องกันกำจัดเมื่อพบใบมะม่วงที่มีอาการโรคบั่วปมใบมะม่วง ควรเด็ดหรือตัดทำลาย รวบรวมใบที่เป็นโรคเผาทำลายและฉีดพ่นป้องกันด้วยสารป้องกันกำจัดเชื้อราหรือน้ำหมักสมุนไพรกำจัดเชื้อรา\nวิธีทำน้ำหมักสมุนไพรกำจัดเชื้อรา\nให้นำสมุนไพรเปลือกมังคุด,สารสะเดา,บอระเพ็ด, ขมิ้นที่ได้มา สับให้ละเอียด จากนั้นจึงนำไปเทลงใส่ในถังหมัก คลุกเคล้าให้สมุนไพรเข้ากัน จากนั้นนำน้ำเปล่า เหล้าขาว และน้ำส้มสายชูใส่ลงในถังหมักแล้วคนให้เข้ากัน เมื่อได้ที่จึงปิดฝาถังหมัก ตั้งทิ้งไว้ในที่ร่มแล้วหมักอย่างน้อย 7 วัน ก็สามารถนำน้ำหมักสมุนไพรยับยั้งเชื้อรา (สูตรเร่งด่วน) ไปใช้งาน\nการใช้น้ำหมักสมุนไพรกำจัดเชื้อรา\n1.การป้องกัน\nนำน้ำหมักสมุนไพรยับยั้งเชื้อรา (สูตรเร่งด่วน) จำนวน 30 ซีซี ผสมกับ น้ำ 20 ลิตร แล้วนำไปฉีดพ่นพืชผัก ไม้ผล ทุกๆ 7 วัน ก็จะสามารถยับยั้งเชื้อรา\n2.เมื่อเกิดเชื้อราแล้ว\nนำน้ำหมักสมุนไพรยับยั้งเชื้อรา (สูตรเร่งด่วน) จำนวน 50 ซีซี ผสมกับ น้ำ 20 ลิตร แล้วนำไปฉีดพ่นพืชผัก ไม้ผล โดยฉีดพ่นในส่วนที่เกิดเชื้อราทุกๆ 3 วัน จะสามารถกำจัดและยับยั้งการเกิดเชื้อราได้"
    elif CLASS_NAMES[prediction] == "โรคราแป้ง":
        ans = "โรคราแป้ง\nสาเหตุและอาการ\nสาเหตุเกิดจากเชื้อรา Oidium mangiferae Benthet ทำให้มะม่วงไม่สามารถติดผลได้ สามารถสังเกตเห็นโรคนี้บนใบได้โดยจะพบขุยของเส้นใยสีขาวด้านใต้ใบ ทำให้ใบบิดม้วนงอ\nวิธีรักษา\n1. การกำจัดวัชพืชในแปลงปลูกในช่วงแรกของการเริ่มฤดูกาลใหม่ เพื่อใส่ปุ๋ยกระตุ้นกิ่งหรือต้นให้สะสมอาหาร สามารถลดปริมาณการเกิดโรคได้\n2. การตัดแต่งกิ่งที่เป็นโรค ตัดกิ่งย่อยที่อยู่ในทรงพุ่ม และตัดแต่งเพื่อสร้างกิ่งใหม่ สร้างช่อดอกใหม่ ทำให้ทรงพุ่มโปร่งความชื้นในทรงพุ่มลดลง ลดความเสียหายจากการเกิดโรค\n3. การใช้ชีวภัณฑ์ฉีดพ่นเป็นประจำเพื่อป้องกันการเกิดโรคราแป้ง เช่น เชื้อราไตรโคเดอร์มา หรือเชื้อแบคทีเรียบาซิลัส ซับทิลิส"
    elif CLASS_NAMES[prediction] == "โรคราดำมะม่วง":
        ans = "โรคราดำมะม่วง\nสาเหตุและอาการ\nสาเหตุเกิดจากเชื้อราหลายๆชนิด แต่ที่พบบ่อยที่สุดเช่น Capnodium sp. และ Meliola sp. โดยมีอาการคราบสีดำปกคลุมบน ลำต้น กิ่ง ใบไม้ ช่อดอก และผลมะม่วง\nวิธีรักษา\n1.ตัดแต่งต้นมะม่วง ตัดแต่งทรงพุ่มให้โปร่ง เพื่อให้การระบายอากาศดี และลดความชื้นในทรงพุ่ม\n2.ป้องกันกำจัดแมลงเพลี้ยจั๊กจั่น เพลี้ยหอยและเพลี้ยแป้ง ไม่ให้ระบาด โดยเฉพาะในช่วงมะม่วงแตกใบอ่อนและแทงช่อดอก จะช่วยไม่ให้เกิดปัญหาราดำรบกวน\n3.หาต้นที่เป็นศูนย์กลางการระบาด เพื่อพ่นสารกำจัดแมลงที่มีประสิทธิภาพต่อเพลี้ยจักจั่นในต้นที่เป็นจุดเริ่มต้นของการระบาดและต้นใกล้เคียงในรัศมีโดยรอบ และอาจจะผสมสารป้องกันกำจัดโรคพืชที่มีประสิทธิภาพร่วมไปด้วยก็ได้\n4.หมั่นตรวจแปลงมะม่วงโดยเฉพาะในช่วงมะม่วงแตกใบอ่อนและแทงช่อดอกอย่างสม่ำเสมอ พ่นสารป้องกันกำจัดเพลี้ยจักจั่น อาจจะสังเกตุได้จากเสียงกระโดดเกาะตามกิ่งและใบของแมลงตัวแก่ทรงพุ่มมะม่วง หรือคราบน้ำหวานที่ดูเหมือนคราบน้ำมันที่เคลือบตามใบ รวมทั้งตรวจดูตัวแมลงตามช่อดอกหรือยอดอ่อนใน\n5.หมั่นตัดแต่งยอดอ่อนที่แตกตามกิ่งก้านในทรงพุ่ม เพื่อไม่ให่เป็นที่อยู่อาศัย และ วางไข่ของแมลง\nคราบราดำที่ปกคลุมส่วนของพืช อาจจะถูกชะล้างออกไปได้บ้าง โดยการพ่นน้ำบ่อยๆ หรืออาจจะล่อนหลุดไปเองตามธรรมชาติในสภาพแห้งแล้งแต่ถ้ามีการระบาดของโรคมากแล้ว ควรผสมสารป้องกันกำจัดโรคพืชที่มีประสิทธิภาพร่วมไปด้วย เช่น \n- สารกลุ่มรหัส 1(เบนโนมิล คาร์เบนดาซิม ไธอะเบนดาโซล ไทโอฟาเนทเมทิล)\n- สารกลุ่มรหัส 3 (ไตรฟอรีน โพรคลอราช ไดฟิโนโคนาโซล อีพ๊อกซีโคนาโซล เฮกซาโคนาโซล ไมโคลบิวทานิล โพรพิโคนาโซล ทีบูโคนาโซล และ เตตราโคนาโซล)\n- สารกลุ่มรหัส 11(อะซ๊อกซีสโตรบิน ไพราโคลสโตรบิน ครีโซซิมเมทิล และ ไตรฟล๊อก)"
    else: ans = "ไม่เป็นโรค"
    delete_pic()
    return ans


@app.route("/webhook", methods=["POST"])
def callback():
    # get X-Line-Signature header value
    signature = request.headers["X-Line-Signature"]

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=(ImageMessage))
def handle_content_message(event):

    ext = "jpg"

    message_content = line_bot_api.get_message_content(event.message.id)

    with tempfile.NamedTemporaryFile(
        dir=static_tmp_path, prefix=ext + "-", delete=False
    ) as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        tempfile_path = tf.name

    dist_path = tempfile_path + "." + ext
    # file name of picture -> jpg-i6h2jbph.jpg
    dist_name = os.path.basename(dist_path)

    image_path = "./tmp/" + dist_name

    os.rename(tempfile_path, dist_path)

    line_bot_api.reply_message(
        event.reply_token,
        [
            TextSendMessage(text="ได้รับรูปภาพแล้ว กำลังวิเคราะห์รูปภาพ"),
            TextSendMessage(text=about_pic(image_path)),
        ],
    )

if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage="Usage: python " + __file__ + " [--port <port>] [--help]"
    )
    arg_parser.add_argument("-p", "--port", default=8000, help="port")
    arg_parser.add_argument("-d", "--debug", default=False, help="debug")
    options = arg_parser.parse_args()

    app.run(debug=options.debug, port=options.port)
