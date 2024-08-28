# BEYOND: BE Your Own NeighborhooD

Deep Neural Networks (DNNs) are vulnerable to Adversarial Examples (AEs), hindering their use in safety-critical systems. In this paper, we present BEYOND, an innovative AE detection framework designed for reliable predictions. BEYOND identifies AEs by distinguishing the AE’s abnormal relation with its augmented versions, i.e. neighbors, from two prospects: representation similarity and label consistency. An off-the-shelf Self-Supervised Learning (SSL) model is used to extract the representation and predict the label for its highly informative representation capacity compared to supervised learning models. We found clean samples maintain a high degree of representation similarity and label consistency relative to their neighbors, in contrast to AEs which exhibit significant discrepancies. We explain this observation and show that leveraging this discrepancy BEYOND can accurately detect AEs. Additionally, we develop a rigorous justification for the effectiveness of BEYOND. Furthermore, as a plug-and-play model, BEYOND can easily cooperate with the Adversarial Trained Classifier (ATC), achieving state-of-the-art (SOTA) robustness accuracy. Experimental results show that BEYOND outperforms baselines by a large margin, especially under adaptive attacks. Empowered by the robust relationship built on SSL, we found that BEYOND outperforms baselines in terms of both detection ability and speed. Project page: https://huggingface.co/spaces/allenhzy/Be-Your-Own-Neighborhood.

# Run the code

## 1. Download Pretrained SSL Model and Target Model

The pretrained SSL model in BEYOND is simsiam, and the weights come from [Solo-learn](https://github.com/vturrisi/solo-learn).
You can download the target model and the SSL model via the [link](https://drive.google.com/drive/folders/1ieEdd7hOj2CIl1FQfu4-3RGZmEj-mesi?usp=sharing)
Or you can train the target model with the following command:

```
python train_resnet18.py --data_dir='your own datasets directory'
```

Put the download or trained weights into the weights directory

## 2. CleanSelect

````
cd AEs
python CleanSelect.py --data_dir='your own datasets directory'
````

## 3. Generate AEs

* FGSM, PGD, CW
```
python attacks.py --attack=fgsm --e=0.05
python attacks.py --attack=pgd
python attacks.py --attack=cw
```

* AutoAttack
```
python auto_attack.py --attack=apgd-ce
```

## 5. Detect

```
python detect_cifar10.py
```

## 6. Adaptive Attack

```
python adaptive_attack.py
python run_orthogonal_pgd.py
```


## Citation
---
If you like or use our work please cite us:
```
@inproceedings{he2022your,
  title={Be Your Own Neighborhood: Detecting Adversarial Examples by the Neighborhood Relations Built on Self-Supervised Learning},
  author={He, Zhiyuan and Yang, Yijun and Chen, Pin-Yu and Xu, Qiang and Ho, Tsung-Yi},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```