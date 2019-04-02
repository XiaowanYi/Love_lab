{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled",
      "version": "0.3.2",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/XiaowanYi/Love_lab/blob/master/warmup1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "metadata": {
        "id": "AjCHnWEooWya",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "import os\n",
        "os.environ[\"CUDA_DEVICE_ORDER\"] = \"PCI_BUS_ID\"\n",
        "os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"0\"\n",
        "import matplotlib\n",
        "matplotlib.use(\"Agg\")\n",
        "\n",
        "import keras\n",
        "from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions\n",
        "from keras.preprocessing import image\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "model = VGG16(weights = 'imagenet')\n",
        "\n",
        "val_path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val/'\n",
        "n_val = len(os.listdir(val_path))\n",
        "print('num of validation files: ', n_val)\n",
        "\n",
        "classes = os.listdir(val_path)\n",
        "val_path_list = []\n",
        "for class_file in classes:\n",
        "  path = val_path + class_file + '/'\n",
        "  images_files = os.listdir(path)\n",
        "  for image_file in images_files:\n",
        "    val_path_list.append(path+'image_file')\n",
        "  \n",
        "def get_prediction(file_path):\n",
        "  img = image.load_img(file_path, target_size=(224, 224))\n",
        "  x = image.img_to_array(img)\n",
        "  x = np.expand_dims(x, axis=0)\n",
        "  preds = model.predict(preprocess_input(x))\n",
        "  return decode_predictions(preds, top=5)[0]\n",
        "\n",
        "results = []\n",
        "for i in range(len(val_path_list)):\n",
        "  if i%1000 == 0:\n",
        "    print ('now predicting', i, 'th image')\n",
        "  results.append(get_prediction(val_path_list[i]))\n",
        "\n",
        "save_path = '/home/xiao/warmup1/'\n",
        "df = pd.DataFrame(results)\n",
        "df.to_csv(save_path)\n",
        "print ('YEAH!')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "V0zBlp9pFzRU",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}