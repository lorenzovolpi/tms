# Papers

## Class imabalancing

### Code

```python
img_num_per_cls = [
    int(img_max * (1 / imb_factor) ** (i / (cls_num - 1)))
    for i in range(cls_num)
]
```

### Formula

$$ n_c = n_max \dot \left( \frac{1}{\mathrm{IMB}} \right)^{\frac{c}{C-1}} $$

### [Large-Scale Long-Tailed Recognition in an Open World (Liu et al., CVPR 2019)](https://arxiv.org/abs/1904.05160)

- gli autori dicono esplicitamente che ImageNet-LT è costruito
> “by sampling classes following a Pareto (power-law) distribution”

### [Bag of Tricks for Long-Tail Visual Recognition of Animal Species in Camera-Trap Images (He et al., CVPR 2021)](https://arxiv.org/abs/2206.12458)

- Questo paper standardizza CIFAR-10-LT / CIFAR-100-LT
- Introduce IMB factor
- Implementazioni ufficiali usano esattamente questa formula
- La reference migliore da citare se si usa CIFAR-LT

#### Come citare 

> “Following standard practice in long-tailed recognition 
> [Liu et al., 2019; He et al., 2021], we construct long-tailed datasets 
> by sub-sampling classes according to a power-law distribution with 
> imbalance factor IMB. Concretely, the number of samples per class 
> follows a geometric progression between $𝑛_max$ and $𝑛_min$."

### [Class-Balanced Loss Based on Effective Number of Samples (Cui et al., CVPR 2019)](https://arxiv.org/abs/1901.05555)

- Prima di quel paper:
    - l’imbalance era trattato in modo euristico 
    - “long-tail” = poche immagini → peggio

- Cui et al. introducono:
    - il concetto di effective number of samples
    - una relazione non lineare tra numero di esempi e informazione utile
    - una giustificazione matematica del fatto che:
        - raddoppiare i dati di una classe head aiuta molto meno che raddoppiare quelli 
        di una classe tail 

#### Come citare

> “Long-tailed distributions are common in real-world data [Liu et al., 2019], 
> and class imbalance significantly degrades performance [Cui et al., 2019].”
