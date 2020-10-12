from torchvision import datasets, models, transforms
# import mb2
import cv2
import numpy as np
import torch
import requests
import os
from shutil import copyfile

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('model/savetest.pth', map_location=DEVICE)

dir = "example2"


model.eval()
if torch.cuda.is_available():
    model.to('cuda')
    device = 'cuda'
else:
    device = 'cpu'

avg_pr0 = 0
avg_pr0_0 = 0
avg_pr1 = 0
avg_pr1_0 = 0
avg_pr2 = 0
avg_pr2_0 = 0
avg_pr3 = 0
avg_pr3_0 = 0
avg_pr4 = 0
avg_pr4_0 = 0
avg_pr5 = 0
avg_pr5_0 = 0
avg_pr6 = 0
avg_pr6_0 = 0
avg_pr7 = 0
avg_pr7_0 = 0
avg_pr8 = 0
avg_pr8_0 = 0
avg_pr9 = 0
avg_pr9_0 = 0


file_list = os.listdir(dir)
file_list_py = [file for file in file_list if file.endswith(".JPEG")]

for image in file_list_py:
    # os.listdir(path) :지정한 디렉토리 내의 모든 파일과 디렉토리의 리스트를 return
    orig_image1 = cv2.imread("example2/"+ image)
    # 경로의 이미지를 읽어옴
    to_pil = transforms.ToPILImage()
    # Convert a tensor or an ndarray to PIL Image
    orig_image = to_pil(orig_image1)
    # 읽어온 이미지들을 PIL 이미지로 Convert
    trans = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor()
                                   , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image_test = image
    # transforms.Compose를 통해 Resize, ToTensor, Normalize를 시켜줍니다.
    image = trans(orig_image)
    image = image.unsqueeze(0)
    # image들에 차원 추가
    image = image.to(device)
    # 쿠다 존재하면 쿠다사용

    with torch.no_grad():  # 해당 블록을 history 트래킹 하지 않겠다.
        result = model(image)

    print("-" * 50)

    pr = torch.argmax(torch.nn.functional.softmax(result[0], dim=0))
    # 이미지 불러옴, softmax 치수 = 0 이중 가장 최대값
    result1 = torch.nn.functional.softmax(result[0], dim=0)
    round_result = round(float(result1[pr]), 4)

    print(f"conf : {round_result}, result : {pr}")

    src = "example2/" + image_test

    if pr==0:
        dst = 'example2/ballon/'+image_test
        copyfile(src,dst)
        #a = a + 1
        avg_pr0 = round_result
        avg_pr0_0 = avg_pr0_0 + avg_pr0
    elif pr==1:
        #cv2.imwrite('example2/banana/banana_{}.JPEG'.format(b),orig_image1)
        dst = 'example2/banana/' + image_test
        copyfile(src, dst)
        #b = b + 1
        avg_pr1 = round_result
        avg_pr1_0 = avg_pr1_0 + avg_pr1
    elif pr==2:
        #cv2.imwrite('example2/bell/bell_{}.JPEG'.format(c),orig_image1)
        dst = 'example2/bell/' + image_test
        copyfile(src, dst)
        #c = c + 1
        avg_pr2 = round_result
        avg_pr2_0 = avg_pr2_0 + avg_pr2
    elif pr==3:
        #cv2.imwrite('example2/cdplayer/cdplayer_{}.JPEG'.format(d),orig_image1)
        dst = 'example2/cdplayer/' + image_test
        copyfile(src, dst)
        #d = d + 1
        avg_pr3 = round_result
        avg_pr3_0 = avg_pr3_0 + avg_pr3
    elif pr==4:
        #cv2.imwrite('example2/cleaver/cleaver_{}.JPEG'.format(e),orig_image1)
        dst = 'example2/cleaver/' + image_test
        copyfile(src, dst)
        #e = e + 1
        avg_pr4 = round_result
        avg_pr4_0 = avg_pr4_0 + avg_pr4
    elif pr==5:
        #cv2.imwrite('example2/cradle/cradle_{}.JPEG'.format(f),orig_image1)
        dst = 'example2/cradle/' + image_test
        copyfile(src, dst)
        #f = f + 1
        avg_pr5 = round_result
        avg_pr5_0 = avg_pr5_0 + avg_pr5
    elif pr==6:
        #cv2.imwrite('example2/crane/crane_{}.JPEG'.format(g),orig_image1)
        dst = 'example2/crane/' + image_test
        copyfile(src, dst)
        #g = g + 1
        avg_pr6 = round_result
        avg_pr6_0 = avg_pr6_0 + avg_pr6
    elif pr==7:
        #cv2.imwrite('example2/daisy/daisy_{}.JPEG'.format(h),orig_image1)
        dst = 'example2/daisy/' + image_test
        copyfile(src, dst)
        #h = h + 1
        avg_pr7 = round_result
        avg_pr7_0 = avg_pr7_0 + avg_pr7
    elif pr==8:
        #cv2.imwrite('example2/helmet/helmet_{}.JPEG'.format(i),orig_image1)
        dst = 'example2/helmet/' + image_test
        copyfile(src, dst)
        #i = i + 1
        avg_pr8 = round_result
        avg_pr8_0 = avg_pr8_0 + avg_pr8
    elif pr==9:
        #cv2.imwrite('example2/speaker/speaker_{}.JPEG'.format(j),orig_image1)
        dst = 'example2/speaker/' + image_test
        copyfile(src, dst)
        #j = j + 1
        avg_pr9 = round_result
        avg_pr9_0 = avg_pr9_0 + avg_pr9

print(f"ballon conf : {avg_pr0_0/113}")
print(f"banana conf : {avg_pr1_0/84}")
print(f"bell conf : {avg_pr2_0/85}")
print(f"cdplayer conf : {avg_pr3_0/99}")
print(f"cleaver conf : {avg_pr4_0/105}")
print(f"cradle conf : {avg_pr5_0/100}")
print(f"crane conf : {avg_pr6_0/104}")
print(f"daisy conf : {avg_pr7_0/103}")
print(f"helmet conf : {avg_pr8_0/101}")
print(f"speaker conf : {avg_pr9_0/106}")