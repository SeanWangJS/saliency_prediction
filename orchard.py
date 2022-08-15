import os
import cv2
import argparse
import glob
import zipfile

varity_filename_dict={
    "192.168.1.101-left": "baimei",
    "192.168.1.101-right": "feixue", 
    "192.168.1.103-left": "mingzhuerhao", 
    "192.168.1.103-right": "mingzhusanhao", 
    "192.168.1.105-left": "mingzhuwuhao", 
    "192.168.1.105-right": "xiangfei", 
    "192.168.1.106-left": "yutu",
    "192.168.1.106-right": "yuxierhao", 
    "192.168.1.108-left": "yuxiyihao", 
    "192.168.1.108-right": "zhongguohong"
}

variety_zh_dict={
    "yuxiyihao": "玉溪一号",
    "baimei": "白美",
    "feixue": "飞雪", 
    "mingzhuerhao": "明珠二号", 
    "mingzhusanhao": "明珠三号", 
    "mingzhuwuhao": "明珠五号", 
    "xiangfei": "香妃", 
    "yutu": "玉兔",
    "yuxierhao": "玉溪二号", 
    "yuxiyihao": "玉溪一号", 
    "zhongguohong": "中国红"
}


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_file", dest="zip_file", help="准备处理的图片压缩文件路径")
    parser.add_argument("--dest_folder", dest="dest_folder", default="/opt/orchard-server/data", help="展示图片的目标文件夹")

    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()

    zip_file=args.zip_file
    dest_folder=args.dest_folder

    ## 解压图片压缩包
    date=os.path.basename(zip_file)
    i = date.index(".")
    date=date[0:i]
    dir_path=os.path.dirname(zip_file)
    image_folder = dir_path + "/" + date
    if not os.path.isdir(image_folder):
        os.makedirs(image_folder)

    with zipfile.ZipFile(zip_file) as zip_ref:
        zip_ref.extractall(image_folder)

    ## 抽帧图片数组
    paths=glob.glob(f"{image_folder}/*.jpg")
    paths=filter(lambda p: p.endswith("_0.jpg"), paths)
    paths=list(paths)

    filenames = []
    ip_path_dict={}
    for p in paths:
        ip=os.path.basename(p).split("_")[0]
        ip_path_dict[ip] = p

    sqls = []
    for ip in ip_path_dict:
        path = ip_path_dict[ip]
        image=cv2.imread(path)
        h, w, _ = image.shape
        left=image[:, 0:w // 2, :]
        right = image[:, w//2:w, :]
        if ip + "-left" not in varity_filename_dict.keys():
            continue
        name1 = varity_filename_dict[f"{ip}-left"]
        name2 = varity_filename_dict[f"{ip}-right"]
        filename_left = name1 + "-" +date + ".jpg"
        filename_right = name2 + "-" +date + ".jpg"

        zh_name1=variety_zh_dict[name1]
        zh_name2=variety_zh_dict[name2]

        sql1='INSERT INTO discern_result2 (variety_name, filename) VALUES ("{}", "{}");'.format(zh_name1, filename_left)
        sql2='INSERT INTO discern_result2 (variety_name, filename) VALUES ("{}", "{}");'.format(zh_name2, filename_right)

        sqls.append(sql1)
        sqls.append(sql2)

        cv2.imwrite(f"{dest_folder}/{filename_left}", left)
        cv2.imwrite(f"{dest_folder}/{filename_right}", right)

    for sql in sqls:
        print(sql)
