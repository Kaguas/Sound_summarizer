import whisper
import numpy as np
import os
import sys
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import json
import torch
import spacy
import time
import re

#path
correct_file = "your wrong word file"



def args_check():
    args = sys.argv
    if len(args) == 1:
        print("please select file\n")
        sys.exit()
    else:
        for items in args:
            if type(items) != str:
                print("please send me by string\n")
                sys.exit()
    
    print("args don't have any problem\n")
    return args

def transcribe_def(input: str):
    model = whisper.load_model("medium")
    result = model.transcribe(input,
                            verbose=True,
                            language="ja")

    #transcript = result["text"]
    return result

def format_modifier(result, date: str, output_name: str):
    torch.cuda.empty_cache()  # 未使用のGPUメモリを解放
    torch.cuda.ipc_collect()  # 共有メモリを解放
    time.sleep(5)
    
    #with open('/home/kagawa/test/segmented_okamura.txt', 'w') as f:
    #    f.write(segmented_trans)
    
    space_match = re.compile(pattern="[\s\t\n\r]+")
    goshicho_match = re.compile(pattern="ご視聴ありがとうございました")
    
    segmented_trans = "動画の日付:" + date + "\n"
    segmented_trans_before_name_delete = "動画の日付:" + date + "\n"
    prev_conv = ""
    modified_name = ""
    
    for res in result["segments"]:
        modified_name = res["text"]
        not_modified_name = res["text"]

        
        #同じ単語の除去とご視聴、スペース、無の除去
        if (prev_conv != "") & (prev_conv == res["text"]):
            print("same word.")
        elif (space_match.fullmatch(string=modified_name) != None):
            print("only space.")
        elif (goshicho_match.search(string=modified_name) != None):
            print("delete ご視聴.") 
        elif (modified_name == ""):
            print("nothing, delete.")
        else:
            segmented_trans = segmented_trans + f'時間:{str(int(res["start"]/60)).zfill(2)}:{str(int(res["start"])%60).zfill(2)} ## 発言:{modified_name}' + "\n"
            segmented_trans_before_name_delete = segmented_trans_before_name_delete + f'時間:{str(int(res["start"]/60)).zfill(2)}:{str(int(res["start"])%60).zfill(2)} ## 発言:{not_modified_name}' + "\n"
            prev_conv = res["text"]
        
    #ファイル書き込み
    print(segmented_trans)
    print(segmented_trans_before_name_delete)
    with open(output_name, 'w') as f:
        f.write(segmented_trans)
    #with open('/home/kagawa/test/segmented_files/segmented_transcript_before_large_' + date + ".txt", 'w') as f:
    #    f.write(segmented_trans_before_name_delete)
        
    json_open = open(correct_file, 'r')
    wrong_words = json.load(json_open)
    
    for correct, wrong in wrong_words.items():
    #print(correct, wrong)
        for item in wrong:
            segmented_trans = segmented_trans.replace(item, correct)
    #segmented_trans = segmented_trans.replace("\n", "<SEPA>")
    return segmented_trans


#gpt部分
class mini_gpt():
    def __init__(self):
        self.user_prompt = """
        ###Request###
        このデータは、whisperを用いて授業の動画を文字起こししたものです。それぞれのデータは、最初の行に録画された日付<YYYY/MM/DD>が入り、その後の行は
        時間:<mm:ss> ## 発言:<発言>
        という形式に沿っています。これらのデータを以下の手順に沿って処理してください。
        
        1. 動画をテーマごとのチャプターに分けてください。
        2. それぞれのチャプターごとに120文字程度で具体的に内容を要約してください。
        3. 録画された<日付(YYYY/MM/DD)>, それぞれのチャプターの<開始時間(mm:ss)>, チャプターの<要約>を「###Example###」を参考にまとめてください。
        
        #注意#
        ・「###Example###」のとおりに出力してください。
        ・文字に装飾はつけずに、「###Example###」のとおりにそのまま出力してください。

        ###Example###
        チャプター1 「タイトル」
        日付:<日付(YYYY/MM/DD)>
        内容:<要約>
        開始時間: <開始時間(mm:ss)>
        <SEPA>

        チャプター2 「タイトル」
        日付:<日付(YYYY/MM/DD)>
        内容:<要約>
        開始時間: <開始時間(mm:ss)>
        <SEPA>

        チャプター3 「タイトル」
        日付:<日付(YYYY/MM/DD)>
        内容:<要約>
        開始時間: <開始時間(mm:ss)>
        <SEPA>
        """
        self.client = OpenAI(api_key = os.environ["GPTKEY"])
    
    def ask(self, segmented_text: str):
        response = self.client.chat.completions.create(
        #model="gpt-3.5-turbo-0125",
        model="gpt-4.1-mini",
        messages=[
                {"role": "assistant", "content": segmented_text},
                {"role": "user", "content": self.user_prompt}
            ],
        )
        return response.choices[0].message.content



if __name__ == '__main__':
    #args[1]はファイル名、args[2]は出力ファイル名
    gpt = mini_gpt()
    args = args_check()
    date_match = re.compile(pattern="20[0-9][0-9][01][0-9][0-3][0-9]")
    after_embedding = [[],[]]
    print(args)
    
    #日付の取得
    result = date_match.search(args[1])
    print(result)
    date_val = args[1][result.start():result.end()]
    print(f"抜き出した日付は{date_val}")
    
    start = time.time()
    
    transcript = transcribe_def(args[1])
    print(transcript)
    segmented_trans = format_modifier(result=transcript, date=date_val, output_name = args[2])
    print(segmented_trans)
    
    result = gpt.ask(segmented_text=segmented_trans)
    print(result)
    
    end = time.time()
    print(f"処理時間は{end - start}秒です。")
    
    with open("transcript_text_" + args[2], 'w') as f:
        f.write(result)
    print("テキストファイルとして保存しました。")
    
    
    
