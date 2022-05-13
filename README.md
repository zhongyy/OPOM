# One Person One Mask (OPOM) 
Code and datasets of TPAMI 2022 paper <OPOM: Customized Invisible Cloak towards Face Privacy Protection>.

While convenient in daily life, face recognition technologies also raise privacy concerns for regular users on the social media since they could be used to analyze face images and videos, efficiently and surreptitiously without any security restrictions. We investigate the face privacy protection from a technology standpoint based on a new type of customized cloak, which can be applied to all the images of a regular user, to prevent malicious face recognition systems from uncovering their identity. Specifically, we propose a new method, named one person one mask (OPOM), to generate person-specific (class-wise) universal masks by optimizing each training sample in the direction away from the feature subspace of the source identity. The effectiveness of the proposed method is evaluated on both common and celebrity datasets against black-box face recognition models. 

![arch](https://github.com/zhongyy/OPOM/blob/main/illustration.jpg)

## Usage Instructions

### Environment
Install Anaconda, Pytorch and MxNet. For other libs, please refer to the file requirements.txt.

```
conda create -n OPOM python=3.7
conda activate OPOM
git clone https://github.com/zhongyy/OPOM.git
pip install -r requirements.txt
```

### Datasets and face recognition models
- Please download Privacy-Commons dataset [Baidu Netdisk](https://pan.baidu.com/s/1djkvaDghom8U7-Y_Nt95uA)(password: 3g2b), [Google Drive](https://drive.google.com/file/d/1NLKVDA-PRNJECtad5qcoDKsL4f1eWJNQ/view?usp=sharing); and Privacy-Celebrities dataset [Baidu Netdisk](https://pan.baidu.com/s/16bWSdmHV8QETLj20ArPnEw)(password: 28cq), [Google Drive](https://drive.google.com/file/d/1AGkA2S9-9zTPue8wZo0kuJ9-B9RAnaP7/view?usp=sharing). 

- Create a folder ['data/'] at the same level with ['code/'], and then unzip the datasets into it. 

- Please download Surrogate models and Target models: [Baidu Netdisk](https://pan.baidu.com/s/1aV1NymYW_L50ECiwJAylMA)(password: y1cy), [Google Drive](https://drive.google.com/file/d/1XmHD2mTcc6SHutCVPVw7cVg5jGkGIIUU/view?usp=sharing).

- Create a folder ['models/'] at the same level with ['code/'], and then unzip the models into it. 

### Privacy Mask Generation
To generate privacy masks of Privacy-Commons dataset, based on surrgate model "Resnet50-WebFace-Softmax", with different approximation methods, and transferability enhancement methods, please do as follows. Other surrogate models can be used modifying "--pretrained". Other parameters, please refer to the code. 
```
cd code/generation
./gen_privacy_common_softmax.sh 
```

To generate privacy masks of Privacy-Celebrities dataset, please do as follows.
```
./gen_privacy_celeb_softmax.sh 
```

### Privacy Mask Evaluation
After generating the privacy masks, please refer to the evaluation part for privacy pretection rate. You can modify "--msk_dir" for different versions of masks. For Privacy-Commons dataset, evaluation towards six target models is as follows.
```
cd code/evaluation
./test_common.sh 
```
For Privacy-Celebrities dataset, evaluation towards six target models is as follows.
```
./test_celeb.sh 
```

## Citation

If you find **OPOM** useful in your research, please consider to cite:

	@ARTICLE{zhong2022OPOM,
	  title = {OPOM: Customized Invisible Cloak towards Face Privacy Protection},
	  author = {Zhong, Yaoyao and Deng, Weihong},
	  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
	  year = {2022}
	}
