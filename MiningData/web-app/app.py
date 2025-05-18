"""
Created on Sun Apr 27 08:38:46 2025

@author: tianr
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)  # 初始化APP


@app.route("/")  # 装饰器
def home():
    return render_template("index.html")  # 先引入index.html，同时根据后面传入的参数，对html进行修改渲染。


@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]  # 存储用户输入的参数
    print(data)
    model_type = data[len(data) - 1]
    data = data[:-1]
    print(data)
    data = np.array(data)  # 将用户输入的值转化为一个数组
    data = [data]
    print(data)

    match model_type:
        case 1:
            model = pickle.load(open("forest-model.pkl", "rb"))
            scaler = pickle.load(open("forest-scaler.pkl", "rb"))
        case 2:
            model = pickle.load(open("mlp-model.pkl", "rb"))
            scaler = pickle.load(open("mlp-scaler.pkl", "rb"))

    data = scaler.transform(data)
    print(data)
    prediction = model.predict(data)  # 输入模型进行预测

    output = prediction[0]  # 将预测值传入output
    print(output)

    extra_text = ''
    match output:
        case 1:
            extra_text = '0-3年'
        case 2:
            extra_text = '3-5年'
        case 3:
            extra_text = '5-10年'
        case 4:
            extra_text = '10年以上'

    return render_template(
        "index.html", prediction_text="患者预期的生存期为: {}".format(extra_text)  # 将预测值返回到Web界面，使我们看到
    )


if __name__ == "__main__":
    app.run(debug=True)  # 调试模式下运行文件，实时反应结果。仅限测试使用，生产模式下不要使用
