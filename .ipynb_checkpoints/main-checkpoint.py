{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "df1fdbee-a1c8-4ec5-a395-baae0dd3ddc1",
   "metadata": {
    "editable": true,
    "slideshow": {
     "slide_type": ""
    },
    "tags": []
   },
   "source": [
    "***Import Libraries***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "7e6fa8f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "import gc\n",
    "from pathlib import Path\n",
    "\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torchaudio\n",
    "from tqdm import tqdm\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from matplotlib import  pyplot as plt\n",
    "from pypesq import pesq\n",
    "import metriccs \n",
    "\n",
    "%matplotlib inline\n",
    "\n",
    "warnings.filterwarnings(action='ignore', category=DeprecationWarning)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "14cb1085",
   "metadata": {},
   "source": [
    "***Set noise class to train model***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "596942f0-603c-4214-a5a3-2f8c7f02aa91",
   "metadata": {},
   "outputs": [],
   "source": [
    "noise_class=3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e4e123f9-e894-4bb6-a467-f80ac4c8e057",
   "metadata": {},
   "outputs": [],
   "source": [
    "TRAIN_INPUT_DIR = Path('Datasets/US_Class2'+str(noise_class)+'_Train_Input')\n",
    "TRAIN_TARGET_DIR = Path('Datasets/US_Class2'+str(noise_class)+'_Train_Output')\n",
    "TEST_NOISY_DIR = Path('Datasets/US_Class2'+str(noise_class)+'_Test_Input')\n",
    "TEST_CLEAN_DIR = Path('Datasets/clean_testset_wav')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec221452-65cc-4742-bd77-6871d49322c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.random.seed(999)\n",
    "torch.manual_seed(999)\n",
    "\n",
    "train_on_gpu=torch.cuda.is_available()\n",
    "\n",
    "if(train_on_gpu):\n",
    "    print('Training on GPU.')\n",
    "else:\n",
    "    print('No GPU available, training on CPU.')\n",
    "       \n",
    "DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6f4ee461-5cd6-4019-8910-c6b7e8b7dcb6",
   "metadata": {},
   "outputs": [],
   "source": [
    "SAMPLE_RATE = 48000\n",
    "N_FFT = (SAMPLE_RATE * 64) // 1000 \n",
    "HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "188a00b5",
   "metadata": {},
   "source": [
    "***Define SpeechDataSet Class***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "261a7cb8-e105-4f91-9c83-950b6ebe0498",
   "metadata": {},
   "outputs": [],
   "source": [
    "class SpeechDataset(Dataset):\n",
    "    \n",
    "    def __init__(self, noisy_files, target_files, n_fft=64, hop_length=16):\n",
    "        super().__init__()\n",
    "        \n",
    "        self.noisy_files = sorted(noisy_files)\n",
    "        self.target_files = sorted(target_files)\n",
    "        \n",
    "        self.n_fft = n_fft\n",
    "        self.hop_length = hop_length\n",
    "        \n",
    "        self.len_ = len(self.noisy_files)\n",
    "        \n",
    "        self.max_len = 165000\n",
    "     \n",
    "    \n",
    "    def __len__(self):\n",
    "        return self.len_\n",
    "      \n",
    "    def load_sample(self, file):\n",
    "        waveform, _ = torchaudio.load(file)\n",
    "        # print(_)\n",
    "        return waveform\n",
    "  \n",
    "    def __getitem__(self, index):\n",
    "\n",
    "        file_t=self.target_files[index]\n",
    "        file_n=self.noisy_files[index]\n",
    "        file_t=str(file_t)\n",
    "        file_n=str(file_n)\n",
    "        x_target = self.load_sample(file_t)\n",
    "        x_noisy = self.load_sample(file_n)\n",
    "        \n",
    "        x_target = self._prepare_sample(x_target)\n",
    "        x_noisy = self._prepare_sample(x_noisy)\n",
    "        \n",
    "        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft, hop_length=self.hop_length,window=torch.ones(N_FFT, device=DEVICE), normalized=True,return_complex=True)\n",
    "        x_target_stft = torch.stft(input=x_target, n_fft=self.n_fft, hop_length=self.hop_length,window=torch.ones(N_FFT, device=DEVICE), normalized=True,return_complex=True)\n",
    "        # return 0\n",
    "        # print(x_noisy_stft)\n",
    "        return torch.view_as_real(x_noisy_stft), torch.view_as_real(x_target_stft)\n",
    "        \n",
    "    def _prepare_sample(self, waveform):\n",
    "        waveform = waveform.numpy()\n",
    "        # print(waveform.shape)\n",
    "        current_len = waveform.shape[1]\n",
    "        # print(current_len)\n",
    "        output = np.zeros((1, self.max_len), dtype='float32')\n",
    "        # print(output.shape)\n",
    "        output[0, -current_len:] = waveform[0,:self.max_len]\n",
    "        output = torch.from_numpy(output)\n",
    "        # print(output.shape)\n",
    "        return output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "176918c9-d2db-4db1-94b7-f5bd987bab50",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_input_files = sorted(list(TRAIN_INPUT_DIR.rglob('*.wav')))\n",
    "train_target_files = sorted(list(TRAIN_TARGET_DIR.rglob('*.wav')))\n",
    "\n",
    "test_noisy_files = sorted(list(TEST_NOISY_DIR.rglob('*.wav')))\n",
    "test_clean_files = sorted(list(TEST_CLEAN_DIR.rglob('*.wav')))\n",
    "\n",
    "# print(\"No. of Training files:\",len(train_input_files))\n",
    "# print(\"No. of Testing files:\",len(test_noisy_files))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "51483b35",
   "metadata": {},
   "source": [
    "***Make SpeechDataSet Objects for test and train files***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "504be1ac-b4da-46d9-8aac-109d010ff275",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_dataset = SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)\n",
    "train_dataset = SpeechDataset(train_input_files, train_target_files, N_FFT, HOP_LENGTH)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ced01078",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "04efc03b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import librosa"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "16bd470e",
   "metadata": {},
   "outputs": [],
   "source": [
    "spectrogram = torchaudio.transforms.Spectrogram(\n",
    "    n_fft=N_FFT,\n",
    "    win_length=16,\n",
    "    hop_length=HOP_LENGTH,\n",
    "    center=True,\n",
    "    pad_mode=\"reflect\",\n",
    "    power=2.0,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "63ac5e7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_spectrogram(spec, title=None, ylabel=\"freq_bin\", aspect=\"auto\", xmax=None):\n",
    "    fig, axs = plt.subplots(1, 1)\n",
    "    axs.set_title(title or \"Spectrogram (db)\")\n",
    "    axs.set_ylabel(ylabel)\n",
    "    axs.set_xlabel(\"frame\")\n",
    "    im = axs.imshow(librosa.power_to_db(spec), origin=\"lower\", aspect=aspect)\n",
    "    if xmax:\n",
    "        axs.set_xlim((0, xmax))\n",
    "    fig.colorbar(im, ax=axs)\n",
    "    plt.show(block=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "3e085499",
   "metadata": {},
   "outputs": [],
   "source": [
    "# spec=spectrogram(train_dataset.load_sample(\"Datasets/US_Class3_Test_Input/p232_001.wav\"))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "ba6f6e23",
   "metadata": {},
   "outputs": [],
   "source": [
    "# plot_spectrogram(spec[1], title=\"torchaudio\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e65b06f",
   "metadata": {},
   "source": [
    "***Make Dataloader for train and test dataset***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "9b927dd1-77b3-48db-8acf-83c50cc7a4a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_loader=DataLoader(test_dataset,batch_size=1,shuffle=True)\n",
    "train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c640f27c",
   "metadata": {},
   "source": [
    "***Define Convolution Class for Complex number For Down Sampling***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1f5410d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "class CConv2d(nn.Module):\n",
    "\n",
    "    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0):\n",
    "        super().__init__()\n",
    "\n",
    "        self.in_channels=in_channels\n",
    "        self.out_channels=out_channels\n",
    "        self.kernel_size=kernel_size\n",
    "        self.padding=padding\n",
    "        self.stride=stride\n",
    "        self.real_conv= nn.Conv2d(in_channels=self.in_channels,\n",
    "                                  out_channels=self.out_channels,\n",
    "                                  kernel_size=self.kernel_size,\n",
    "                                  padding=self.padding,\n",
    "                                  stride=self.stride)\n",
    "        self.imag_conv = nn.Conv2d(in_channels=self.in_channels,\n",
    "                                  out_channels=self.out_channels,\n",
    "                                  kernel_size=self.kernel_size,\n",
    "                                  padding=self.padding,\n",
    "                                  stride=self.stride)\n",
    "        \n",
    "        nn.init.xavier_uniform_(self.real_conv.weight)\n",
    "        nn.init.xavier_uniform_(self.imag_conv.weight)\n",
    "    def forward(self,x):\n",
    "        # print(x.shape)\n",
    "        x_real = x[...,0]\n",
    "        x_imag= x[...,1]\n",
    "        c_real = self.real_conv(x_real)-self.imag_conv(x_imag)\n",
    "        c_imag=self.imag_conv(x_real)+self.real_conv(x_imag)\n",
    "\n",
    "        output = torch.stack([c_real,c_imag],dim=-1)\n",
    "        return output\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0140778e",
   "metadata": {},
   "source": [
    "***Define Convolution Class for Complex number For Up Sampling***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "516694e8-9dd0-4408-b6da-c7e964113f9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "class CConvTranspose2d(nn.Module):\n",
    "    def __init__(self,in_channels,out_channels,kernel_size,stride,out_padding=0,padding=0):\n",
    "        super().__init__()\n",
    "        self.in_channels=in_channels\n",
    "        self.out_channels=out_channels\n",
    "        self.kernel_size=kernel_size\n",
    "        self.padding=padding\n",
    "        self.out_padding=out_padding\n",
    "        self.stride=stride\n",
    "        \n",
    "        self.real_convt= nn.ConvTranspose2d(in_channels=self.in_channels,\n",
    "                                            out_channels=self.out_channels,\n",
    "                                            kernel_size=self.kernel_size,\n",
    "                                            stride=self.stride,\n",
    "                                            output_padding=self.out_padding,\n",
    "                                            padding=self.padding)\n",
    "        self.imag_convt=nn.ConvTranspose2d(in_channels=self.in_channels,\n",
    "                                            out_channels=self.out_channels,\n",
    "                                            kernel_size=self.kernel_size,\n",
    "                                            stride=self.stride,\n",
    "                                            output_padding=self.out_padding,\n",
    "                                            padding=self.padding)\n",
    "        nn.init.xavier_uniform_(self.real_convt.weight)\n",
    "        nn.init.xavier_uniform_(self.imag_convt.weight)\n",
    "\n",
    "\n",
    "    def forward(self,x):\n",
    "        # print(x.shape)\n",
    "        x_real=x[...,0]\n",
    "        x_imag=x[...,1]\n",
    "\n",
    "        c_real=self.real_convt(x_real)-self.imag_convt(x_imag)\n",
    "        c_imag=self.imag_convt(x_real)+self.real_convt(x_imag)\n",
    "            \n",
    "        output=torch.stack([c_real,c_imag],dim=-1)\n",
    "        \n",
    "        return output"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d7602cb4",
   "metadata": {},
   "source": [
    "***Define Batch Normalisation Class***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "2bc31d53",
   "metadata": {},
   "outputs": [],
   "source": [
    "class CBatchNorm2D(nn.Module):\n",
    "    def __init__(self,num_features,eps=1e-05,momentum=0.1,affine=True, track_running_stats=True):\n",
    "        super().__init__()\n",
    "        self.num_features=num_features\n",
    "        self.affine=affine\n",
    "        self.momentum=momentum\n",
    "        self.eps=eps\n",
    "        self.track_running_stats=track_running_stats\n",
    "\n",
    "        self.real_b=nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,\n",
    "                                      affine=self.affine, track_running_stats=self.track_running_stats)\n",
    "        self.imag_b=nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,\n",
    "                                      affine=self.affine, track_running_stats=self.track_running_stats)\n",
    "        \n",
    "    def forward(self,x):\n",
    "        real=x[...,0]\n",
    "        imag=x[...,1]\n",
    "\n",
    "        real_cb=self.real_b(real)\n",
    "        imag_cb=self.imag_b(imag)\n",
    "\n",
    "        output=torch.stack([real_cb,imag_cb],dim=-1)\n",
    "        # print(x.shape)\n",
    "\n",
    "        return output\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "80f51604",
   "metadata": {},
   "source": [
    "***Encoder***\n",
    "\n",
    "Convolution --> Complex Batch Normalisation ---> Activation Function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "4aed2897",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Encoder(nn.Module):\n",
    "    def __init__(self,in_channels=1,out_channels=45,stride=(2,2),kernel_size=(7,5),padding=(0,0)):\n",
    "        super().__init__()\n",
    "        self.kernel_size=kernel_size\n",
    "        self.stride=stride\n",
    "        self.padding=padding\n",
    "        self.in_channels=in_channels\n",
    "        self.out_channels=out_channels\n",
    "\n",
    "        self.cconv= CConv2d(in_channels=self.in_channels,\n",
    "                            out_channels=self.out_channels,\n",
    "                            kernel_size=self.kernel_size,\n",
    "                            stride=self.stride,\n",
    "                            padding=self.padding)\n",
    "        self.cbn=CBatchNorm2D(num_features=self.out_channels)\n",
    "\n",
    "        self.leaky_relu=nn.LeakyReLU()\n",
    "        \n",
    "    def forward(self,x):\n",
    "        conved=self.cconv(x)\n",
    "        normed=self.cbn(conved)\n",
    "        acted=self.leaky_relu(normed)\n",
    "        # print(acted.shape)\n",
    "        return acted\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "79146932",
   "metadata": {},
   "source": [
    "***Decoder***\n",
    "\n",
    "Transposed Convolution --->  Complex Batch Normalisation --->  Activation Function "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "e103f16b",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Decoder(nn.Module):\n",
    "    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,out_padding,last_layer):\n",
    "        super().__init__()\n",
    "\n",
    "        self.kernel_size = kernel_size\n",
    "        self.stride_size = stride\n",
    "        self.in_channels = in_channels\n",
    "        self.out_channels = out_channels\n",
    "        self.out_padding = out_padding\n",
    "        self.padding = padding\n",
    "        \n",
    "        self.last_layer = last_layer\n",
    "        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, \n",
    "                                       out_channels=self.out_channels, \n",
    "                             kernel_size=self.kernel_size,\n",
    "                               stride=self.stride_size, \n",
    "                               out_padding=self.out_padding, \n",
    "                               padding=self.padding\n",
    "                               )\n",
    "        \n",
    "        self.cbn = CBatchNorm2D(num_features=self.out_channels) \n",
    "        \n",
    "        self.leaky_relu = nn.LeakyReLU()\n",
    "            \n",
    "    def forward(self, x):\n",
    "        \n",
    "        conved = self.cconvt(x)\n",
    "        \n",
    "        if not self.last_layer:\n",
    "            normed = self.cbn(conved)\n",
    "            output = self.leaky_relu(normed)\n",
    "        else:\n",
    "            m_phase = conved / (torch.abs(conved) + 1e-8)\n",
    "            m_mag = torch.tanh(torch.abs(conved))\n",
    "            output = m_phase * m_mag\n",
    "            \n",
    "        return output"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d8cafa36",
   "metadata": {},
   "source": [
    "***DCUNET Model Class***\n",
    "\n",
    "- Define Parameters\n",
    "- Creates Encoders For Down Sampling\n",
    "- Creates Decoders For Up Sampling\n",
    "- Apply Down Sampling and then UpSampling adn return istft Output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "73f37211",
   "metadata": {},
   "outputs": [],
   "source": [
    "class DCUnet(nn.Module):\n",
    "    \"\"\"\n",
    "    Deep Complex U-Net class of the model.\n",
    "    \"\"\"\n",
    "    def __init__(self, n_fft=64, hop_length=16):\n",
    "        super().__init__()\n",
    "        \n",
    "        # for istft\n",
    "        self.n_fft = n_fft\n",
    "        self.hop_length = hop_length\n",
    "        \n",
    "        self.set_size(model_complexity=int(45//1.414), input_channels=1, model_depth=20)\n",
    "        self.encoders = []\n",
    "        self.model_length = 20 // 2\n",
    "        \n",
    "        for i in range(self.model_length):\n",
    "            module = Encoder(in_channels=self.enc_channels[i], out_channels=self.enc_channels[i + 1],\n",
    "                             kernel_size=self.enc_kernel_sizes[i], stride=self.enc_strides[i], padding=self.enc_paddings[i])\n",
    "            self.add_module(\"encoder{}\".format(i), module)\n",
    "            self.encoders.append(module)\n",
    "\n",
    "        self.decoders = []\n",
    "\n",
    "        for i in range(self.model_length):\n",
    "            if i != self.model_length - 1:\n",
    "                module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_length - i], out_channels=self.dec_channels[i + 1], \n",
    "                                 kernel_size=self.dec_kernel_sizes[i], stride=self.dec_strides[i], padding=self.dec_paddings[i],\n",
    "                                 out_padding=self.dec_output_padding[i],last_layer=False)\n",
    "            else:\n",
    "                module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_length - i], out_channels=self.dec_channels[i + 1], \n",
    "                                 kernel_size=self.dec_kernel_sizes[i], stride=self.dec_strides[i], padding=self.dec_paddings[i],\n",
    "                                 out_padding=self.dec_output_padding[i], last_layer=True)\n",
    "            self.add_module(\"decoder{}\".format(i), module)\n",
    "            self.decoders.append(module)\n",
    "       \n",
    "        \n",
    "    def forward(self, x, is_istft=True):\n",
    "        orig_x = x\n",
    "        xs = []\n",
    "        for i, encoder in enumerate(self.encoders):\n",
    "            xs.append(x)\n",
    "            x = encoder(x)\n",
    "        p = x\n",
    "        for i, decoder in enumerate(self.decoders):\n",
    "            p = decoder(p)\n",
    "            if i == self.model_length - 1:\n",
    "                break\n",
    "            p = torch.cat([p,xs[self.model_length - 1 - i]], dim=1)\n",
    "        mask = p\n",
    "        output = mask * orig_x\n",
    "        output = torch.squeeze(output, 1)\n",
    "        output=torch.view_as_complex(output)\n",
    "        if is_istft:\n",
    "            output = torch.istft(output,n_fft=self.n_fft, hop_length=self.hop_length,\n",
    "                                 window=torch.ones(N_FFT, device=DEVICE),return_complex=False, \n",
    "                                 normalized=True)\n",
    "        return output\n",
    "\n",
    "    \n",
    "    def set_size(self, model_complexity, model_depth=20, input_channels=1):\n",
    "\n",
    "        if model_depth == 20:\n",
    "            self.enc_channels = [input_channels,\n",
    "                                 model_complexity,\n",
    "                                 model_complexity,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity * 2,\n",
    "                                 128]\n",
    "\n",
    "            self.enc_kernel_sizes = [(7, 1),\n",
    "                                     (1, 7),\n",
    "                                     (6, 4),\n",
    "                                     (7, 5),\n",
    "                                     (5, 3),\n",
    "                                     (5, 3),\n",
    "                                     (5, 3),\n",
    "                                     (5, 3),\n",
    "                                     (5, 3),\n",
    "                                     (5, 3)]\n",
    "\n",
    "            self.enc_strides = [(1, 1),\n",
    "                                (1, 1),\n",
    "                                (2, 2),\n",
    "                                (2, 1),\n",
    "                                (2, 2),\n",
    "                                (2, 1),\n",
    "                                (2, 2),\n",
    "                                (2, 1),\n",
    "                                (2, 2),\n",
    "                                (2, 1)]\n",
    "\n",
    "            self.enc_paddings = [(3, 0),\n",
    "                                 (0, 3),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0)]\n",
    "\n",
    "            self.dec_channels = [0,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity * 2,\n",
    "                                 model_complexity,\n",
    "                                 model_complexity,\n",
    "                                 1]\n",
    "\n",
    "            self.dec_kernel_sizes = [(6, 3), \n",
    "                                     (6, 3),\n",
    "                                     (6, 3),\n",
    "                                     (6, 4),\n",
    "                                     (6, 3),\n",
    "                                     (6, 4),\n",
    "                                     (8, 5),\n",
    "                                     (7, 5),\n",
    "                                     (1, 7),\n",
    "                                     (7, 1)]\n",
    "\n",
    "            self.dec_strides = [(2, 1), \n",
    "                                (2, 2), \n",
    "                                (2, 1), \n",
    "                                (2, 2), \n",
    "                                (2, 1), \n",
    "                                (2, 2), \n",
    "                                (2, 1), \n",
    "                                (2, 2), \n",
    "                                (1, 1),\n",
    "                                (1, 1)]\n",
    "\n",
    "            self.dec_paddings = [(0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 0),\n",
    "                                 (0, 3),\n",
    "                                 (3, 0)]\n",
    "            \n",
    "            self.dec_output_padding = [(0,0),\n",
    "                                       (0,0),\n",
    "                                       (0,0),\n",
    "                                       (0,0),\n",
    "                                       (0,0),\n",
    "                                       (0,0),\n",
    "                                       (0,0),\n",
    "                                       (0,0),\n",
    "                                       (0,0),\n",
    "                                       (0,0)]\n",
    "        else:\n",
    "            raise ValueError(\"Unknown model depth : {}\".format(model_depth))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "bf9ea54d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "basepath = str(noise_class)+\"NoiseToNoise\"\n",
    "os.makedirs(basepath,exist_ok=True)\n",
    "os.makedirs(basepath+\"/Weights\",exist_ok=True)\n",
    "os.makedirs(basepath+\"/Samples\",exist_ok=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bcc2956f",
   "metadata": {},
   "source": [
    "***Loss Function***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "48a1ec00",
   "metadata": {},
   "outputs": [],
   "source": [
    "def wsdr_fn(x_, y_pred_, y_true_, eps=1e-8):\n",
    "   \n",
    "    y_true_ = torch.squeeze(y_true_, 1)\n",
    "    y_true = torch.istft(torch.view_as_complex(y_true_), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=torch.ones(N_FFT, device=DEVICE),return_complex=False)\n",
    "    x_ = torch.squeeze(x_, 1)\n",
    "    x = torch.istft(torch.view_as_complex(x_), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=torch.ones(N_FFT, device=DEVICE),return_complex=False)\n",
    "\n",
    "    y_pred = y_pred_.flatten(1)\n",
    "    y_true = y_true.flatten(1)\n",
    "    x = x.flatten(1)\n",
    "\n",
    "\n",
    "    def sdr_fn(true, pred, eps=1e-8):\n",
    "        num = torch.sum(true * pred, dim=1)\n",
    "        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)\n",
    "        return -(num / (den + eps))\n",
    "\n",
    "\n",
    "    z_true = x - y_true\n",
    "    z_pred = x - y_pred\n",
    "\n",
    "    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)\n",
    "    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)\n",
    "    return torch.mean(wSDR)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50407edb",
   "metadata": {},
   "source": [
    "***Train Epoch***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "11f1f7cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_epoch(net, train_loader, loss_fn, optimizer):\n",
    "    net.train()\n",
    "    train_ep_loss = 0.\n",
    "    counter = 0\n",
    "    try:\n",
    "        for noisy_x, clean_x in train_loader:\n",
    "\n",
    "            noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)\n",
    "\n",
    "            net.zero_grad()\n",
    "\n",
    "            pred_x = net(noisy_x)\n",
    "            loss = loss_fn(noisy_x, pred_x, clean_x)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "\n",
    "            train_ep_loss += loss.item() \n",
    "            counter += 1\n",
    "            # print(loss)\n",
    "            # if counter>=1:\n",
    "                # break\n",
    "\n",
    "        train_ep_loss /= counter\n",
    "\n",
    "        gc.collect()\n",
    "        torch.cuda.empty_cache()\n",
    "        return train_ep_loss\n",
    "    except:\n",
    "        train_ep_loss"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fbe5e6c0",
   "metadata": {},
   "source": [
    "***Test Epoch***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "658b1263",
   "metadata": {},
   "outputs": [],
   "source": [
    "def test_epoch(net, test_loader, loss_fn):\n",
    "    try:    \n",
    "        net.eval()\n",
    "        test_ep_loss = 0.\n",
    "        counter = 0.\n",
    "    \n",
    "        for noisy_x, clean_x in test_loader:\n",
    "        \n",
    "\n",
    "            noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)\n",
    "            pred_x = net(noisy_x)\n",
    "       \n",
    "            loss = loss_fn(noisy_x, pred_x, clean_x)\n",
    "\n",
    "            test_ep_loss += loss.item() \n",
    "        \n",
    "            counter += 1\n",
    "            # print(loss)\n",
    "            # print(loss.item())\n",
    "            #for checking the working of Model\n",
    "            # if counter>=1:\n",
    "            #     break\n",
    "     \n",
    "        test_ep_loss /= counter\n",
    "    \n",
    "    \n",
    "        gc.collect()\n",
    "        torch.cuda.empty_cache()\n",
    "        \n",
    "    \n",
    "        return test_ep_loss\n",
    "    except Exception as error:\n",
    "        print(error)\n",
    "        test_ep_loss,{}\n",
    "        "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7926ab42",
   "metadata": {},
   "source": [
    "***Train***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "2ecf04cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train(net, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs):\n",
    "    train_losses = []\n",
    "    test_losses = []\n",
    "    try:\n",
    "        \n",
    "    # print(\"eme\")\n",
    "        for e in tqdm(range(epochs)):\n",
    "        \n",
    "        # print(\"sms\")\n",
    "            # if e == 0:\n",
    "            #     with torch.no_grad():\n",
    "            #         test_loss = test_epoch(net, test_loader, loss_fn)\n",
    "                \n",
    "            #     print(\"Loss before training:{:.6f}\".format(test_loss))\n",
    "            train_loss = train_epoch(net, train_loader, loss_fn, optimizer)\n",
    "            test_loss = 0\n",
    "            scheduler.step()\n",
    "        \n",
    "            with torch.no_grad():\n",
    "                test_loss = test_epoch(net, test_loader, loss_fn)\n",
    "\n",
    "            train_losses.append(train_loss)\n",
    "            test_losses.append(test_loss)\n",
    "            \n",
    "            torch.save(net.state_dict(), basepath +'/Weights/dc20_model_'+str(e+1)+'.pth')\n",
    "            torch.save(optimizer.state_dict(), basepath+'/Weights/dc20_opt_'+str(e+1)+'.pth') \n",
    "        \n",
    "\n",
    "            torch.cuda.empty_cache()\n",
    "            gc.collect()\n",
    "            print(\"Epoch: {}/{}...\".format(e+1, epochs),\n",
    "                      \"Loss: {:.6f}...\".format(train_loss),\n",
    "                      \"Test Loss: {:.6f}\".format(test_loss))\n",
    "            \n",
    "        return train_losses, test_losses\n",
    "    except Exception as error:\n",
    "        print(error) \n",
    "        return train_losses, test_losses"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ef1c6be9",
   "metadata": {},
   "source": [
    "***Create Model Object***\n",
    "\n",
    "With Adam Optimiser"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "4b9ac492",
   "metadata": {},
   "outputs": [],
   "source": [
    "gc.collect()\n",
    "torch.cuda.empty_cache()\n",
    "dcunet20 = DCUnet(N_FFT, HOP_LENGTH).to(DEVICE)\n",
    "optimizer = torch.optim.Adam(dcunet20.parameters())\n",
    "loss_fn = wsdr_fn\n",
    "scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "f9a5c63c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# specify paths and uncomment to resume training from a given point\n",
    "# model_checkpoint = torch.load(path_to_model)\n",
    "# opt_checkpoint = torch.load(path_to_opt)\n",
    "# dcunet20.load_state_dict(model_checkpoint)\n",
    "# optimizer.load_state_dict(opt_checkpoint)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f9984f52",
   "metadata": {},
   "source": [
    "***Train Model***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "c3ed1d1d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# train_losses, test_losses = train(dcunet20, train_loader, test_loader, loss_fn, optimizer, scheduler, 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ce740fe",
   "metadata": {},
   "source": [
    "***Plot Loss Vs Epoch***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "97178e3a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# print(train_losses)\n",
    "# f = plt.figure(figsize=(8, 6))\n",
    "# plt.grid()\n",
    "# plt.plot(test_losses, label='test')\n",
    "# plt.xlabel('epoch')\n",
    "# plt.ylabel('loss')\n",
    "# plt.legend()\n",
    "# plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5aaa1d8c",
   "metadata": {},
   "source": [
    "***Metric***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "5ad6bca1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def pesq_score(net, test_loader):\n",
    "    net.eval()\n",
    "    test_pesq = 0.\n",
    "    counter = 0.\n",
    "\n",
    "\n",
    "    for noisy_x, clean_x in test_loader:\n",
    "        \n",
    "        noisy_x = noisy_x.to(DEVICE)\n",
    "        with torch.no_grad():\n",
    "            pred_x = net(noisy_x)\n",
    "        clean_x = torch.squeeze(clean_x, 1)\n",
    "        clean_x = torch.istft(torch.view_as_complex(clean_x), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=torch.ones(N_FFT, device=DEVICE),return_complex=False)\n",
    "        \n",
    "        \n",
    "        psq = 0.\n",
    "        for i in range(len(clean_x)):\n",
    "            clean_x_16 = torchaudio.transforms.Resample(48000, 16000)(clean_x[i,:].view(1,-1))\n",
    "            pred_x_16 = torchaudio.transforms.Resample(48000, 16000)(pred_x[i,:].view(1,-1))\n",
    "\n",
    "            clean_x_16 = clean_x_16.cpu().cpu().numpy()\n",
    "            pred_x_16 = pred_x_16.detach().cpu().numpy()\n",
    "            \n",
    "            \n",
    "            \n",
    "            psq += pesq(clean_x_16.flatten(), pred_x_16.flatten(), 16000)\n",
    "            \n",
    "        psq /= len(clean_x)\n",
    "        test_pesq += psq\n",
    "        counter += 1\n",
    "\n",
    "    test_pesq /= counter \n",
    "    return test_pesq"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "06a63f9f",
   "metadata": {},
   "source": [
    "***Signal Analysis For Clean, Predicted And Noisy***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "bd518f65",
   "metadata": {},
   "outputs": [],
   "source": [
    "index = 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "6d890e15",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_noisy_files = sorted(list(Path(\"Samples/Sample_Test_Input\").rglob('*.wav')))\n",
    "test_clean_files = sorted(list(Path(\"Samples/Sample_Test_Target\").rglob('*.wav')))\n",
    "\n",
    "test_dataset = SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)\n",
    "\n",
    "# For testing purpose\n",
    "test_loader_single_unshuffled = DataLoader(test_dataset, batch_size=1, shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "47f33767",
   "metadata": {},
   "outputs": [],
   "source": [
    "dcunet20.eval()\n",
    "test_loader_single_unshuffled_iter = iter(test_loader_single_unshuffled)\n",
    "\n",
    "x_n, x_c = next(test_loader_single_unshuffled_iter)\n",
    "for _ in range(index):\n",
    "    x_n, x_c = next(test_loader_single_unshuffled_iter)\n",
    "\n",
    "x_est = dcunet20(x_n, is_istft=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "ca4469e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_est_np = x_est[0].view(-1).detach().cpu().numpy()\n",
    "x_c_np = torch.istft(torch.view_as_complex(torch.squeeze(x_c[0], 1)), n_fft=N_FFT, hop_length=HOP_LENGTH,window=torch.ones(N_FFT, device=DEVICE),return_complex=False, normalized=True).view(-1).detach().cpu().numpy()\n",
    "x_n_np = torch.istft(torch.view_as_complex(torch.squeeze(x_n[0], 1)), n_fft=N_FFT, hop_length=HOP_LENGTH,window=torch.ones(N_FFT, device=DEVICE),return_complex=False, normalized=True).view(-1).detach().cpu().numpy()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "62be4177",
   "metadata": {},
   "source": [
    "***Noisy***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "eecc9338",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1a4dbd9efc0>]"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(x_n_np)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90a201a6",
   "metadata": {},
   "source": [
    "***Predicted***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "62772ea8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1a4dbe12cf0>]"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(x_est_np)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3399114",
   "metadata": {},
   "source": [
    "***Clean***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "9da6413b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1a4d9da3e60>]"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(x_c_np)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c14409bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "metric = metriccs.AudioMetrics(x_c_np, x_est_np, SAMPLE_RATE)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "ebb7affc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CSIG : 1.000\n",
      "CBAK : 1.574\n",
      "COVL : 1.000\n",
      "PESQ : 1.085\n",
      "SSNR : -5.150\n",
      "STOI : 0.804\n",
      "SNR : -2.972\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(metric.display())"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
