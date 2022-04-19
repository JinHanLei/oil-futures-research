import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd

modelused = ['LSTM','RLSTM','GRU','R-GRU']
for mo in modelused:
    model_path = "output/{}_regress.csv".format(mo)
    save_path = './output/{}.png'.format(mo)
    df = pd.read_csv(model_path)

    x = df.index
    y1 = df["predict"]
    y2 = df["label"]

    plt.plot(x, y1, color="blue")
    plt.plot(x, y2, color="red")

    # x刻度间隔
    x_major_locator=MultipleLocator(10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    # 设置标题和字号
    plt.title('regress',fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Price', fontsize=10)

    plt.savefig(save_path)
    plt.legend(['predict', 'label'])
    plt.show()
