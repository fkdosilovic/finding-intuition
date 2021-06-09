---
title: "Merging statistically similar regions"
date: 2021-02-22
categories:
  - blog
tags:
  - computer-vision
  - segmentation
---

Last semester (Fall 2020.) I took a _Computer Vision_ course. The primary reason
I decided to attend the class was the course content. Even though many of the
(online) computer vision courses today, some made available by popular and high
ranking universities, focus on applying deep learning to various computer vision
problems, this class emphasized the traditional approaches to computer vision.
Gaussian and Sobel filter, Hough transform, and various other traditional
approaches, probably forgotten by computer science students around the world,
were the main content of the course. And, I have to say, I enjoyed the class.

Part of the course's grade is obtained through the seminar. The theme of my
seminar was *Merging statistically similar regions*. And, in this blog post,
I'll describe the theory behind the approach.

The rest of this blog post is organized as follows: In the next section,
*Problem formulation*, we formally define the (image) segmentation problem.
Afterwards, we will look at how to merge similar regions, where the notion of
similarity is defined by comparing statistical properties of regions.
The last section is reserved for full derivation of the likelihood ratio.

## Problem formulation

The goal of segmentation is to divide an image into regions, where each region
bears unique semantics - such as car, house, etc. . More formally, we
assume that the image $I$ is **partitioned** into regions

$$
I = \bigcup\limits_{i=1}^{N} R_{i} ,
$$

where each $R_i$, for $i \in \{1, \ldots, N \}$, represents a region.

The goal of segmentation is to find those regions, as they might correspond
to various objects of interest.

Today, state-of-the-art results are achieved by deep neural networks, but,
as it was mentioned in the introduction, we'll look at a more traditional
approach.

## Merging statistically similar regions

Segmentation based on merging (and splitting) [1] follows the approach
presented in the following pseudocode:

```python
# Create initial segmentation.
regions = create_initial_regions(image)

# Prepare RAG.
rag = create_rag(image, regions)

# Merge similar regions.
while at least one pair is merged:
  for region in rag.regions:
    for neighbour_region in rag.neighborhood(region):
      if similar(region, neighbour_region): 
        rag.merge(region, neighbour_region)
```

Region Adjacency Graph (RAG) is a data structure used for manipulating
regions in an image and describe a relationship among them. Since we are not
concerned with details of inner workings of RAGs in this blog post, for more
in depth discussion interested reader is referred to [1].

We are primarily interested in how to define similarity between two regions.

We'll adopt a "probabilitic" approach to image segmentation. We assume that
the pixel values in a region have the same intensity and are corrupted by
zero mean Gaussian noise.

Let $R_1$ and $R_2$ be regions made of $n$ and $m$ pixels, respectively. We'll
model our image segmentation problem in the framework of hypothesis testing:

- $H_0$: pixel values from regions $R_1$ and $R_2$ come from the same distribution
- $H_1$: pixel values from regions $R_1$ and $R_2$ come from different distributions

Our assumption for the null hypothesis is that regions $R_1$ and $R_2$ belong to
the same object, while alternative hypothesis suggests that regions $R_1$ and
$R_2$ belong to different objects.

We can test the hypotheses by comparing ratio of their likelihoods

$$
    L = \frac{p(x^{(1)}, x^{(2)}, \ldots, x^{(n + m)} | H_1)}
            {p(x^{(1)}, x^{(2)}, \ldots, x^{(n + m)} | H_0)} ,
$$

with some predefined threshold. This test is known as the **likelihood ratio test** [2].
If $L \lt T$, where $T$ is a predefined threshold, then there is enough evidence
supporting the null hypothesis, i.e. regions belong to the same object,
otherwise, regions belong to different objects.

Under the null hypothesis, we assume that the regions belong to the same object,
i.e. that the pixels from both regions are sampled from the same distribution

$$
p(x; \mu_0, \Sigma_0) =
\frac{1}{\sqrt{(2\pi)^{\text{D}}|\Sigma_0|}}
\exp{\left\{ -\frac{1}{2}(x - \mu_0)^{\text{T}}\Sigma_0^{-1}
(x - \mu_0) \right\} } ,
$$

where $D = 3$, $x, \mu_0 \in \mathbb{R}^3$ and $\Sigma_0 \in \mathbb{R}^{3 \times 3}$.

The likelihood of obtaining such a sample equals

$$
\begin{align}
p(x^{(1)}, \ldots, x^{(n + m)} | H_0)
\stackrel{\text{i.i.d.}}{=}& \prod_{i=1}^{n + m} p(x^{(i)}|H_0) \\
=& \prod_{i=1}^{n + m} \frac{1}{\sqrt{(2\pi)^3|\Sigma_0|}}
\exp{\left\{-D_M\left(x^{(i)}, \mu_0, \Sigma_0\right)\right\}} \tag{1}\ \\
=& \left[(2\pi)^3|\Sigma_0|\right]^{-\frac{n + m}{2}}
\exp{\left\{-\sum_{i=1}^{n + m}D_M\left(x^{(i)}, \mu_0, \Sigma_0\right)\right\}} ,
\end{align}
$$

where $D_M\left( x, \mu, \Sigma\right) =\frac{1}{2} \left(x - \mu\right)^\text{T}\Sigma^{-1}\left(x - \mu\right)$

After taking care of term inside $\exp$, i.e. the sum of squared Mahalanobis
distances, the likelihood (1) simplifies to

$$
\tag{1a}
p(x^{(1)}, \ldots, x^{(n + m)} | H_0) = \left[(2\pi)^3|\hat{\Sigma}_0|\right]^{-\frac{n + m}{2}}e^{-\frac{3}{2}(n + m)},
$$

where instead of the determinant of the covariance matrix we used the
determinant of the maximum likelihood estimate of $\Sigma_0$.

Similar remarks can be made for the likelihood under alternative hypothesis:

$$
\begin{align}
p(x^{(1)}, \ldots, x^{(n + m)} | H_1)
=& \prod_{i=1}^{n} p(x^{(i)}|H_1) \prod_{j=1}^{m} p(x^{(j)}|H_1) \tag{2} \\
=& \prod_{i=1}^{n} \frac{1}{\sqrt{(2\pi)^3|\Sigma_1|}}
e^{-D_M\left(x^{(i)}, \mu_1, \Sigma_1\right)}
\prod_{j=1}^{m} \frac{1}{\sqrt{(2\pi)^3|\Sigma_2|}}
e^{-D_M\left(x^{(j)}, \mu_2, \Sigma_2\right)}.
\end{align}
$$

Simplifying the likelihood (2) yields

$$
p(x^{(1)}, \ldots, x^{(n + m)} | H_1) = \left[(2\pi)^3|\hat{\Sigma}_1|\right]^{-\frac{n}{2}} e^{-\frac{3}{2}n}
\left[(2\pi)^3|\hat{\Sigma}_2|\right]^{-\frac{m}{2}} e^{-\frac{3}{2}m} \tag{2a}
$$

where $\hat{\Sigma}_1$ and $\hat{\Sigma}_2$ are maximum likelihood estimates of $\Sigma_1$ and $\Sigma_2$, respectively,
for regions $R_1$ and $R_2$, under $H_1$.

Finally, combining (1a) and (2a) into the likelihood ratio yields

$$
\tag{3}
L = \frac{|\hat{\Sigma}_0|^{\frac{n + m}{2}}}{|\hat{\Sigma}_1|^{\frac{n}{2}}|\hat{\Sigma}_2|^{\frac{m}{2}}}
$$

## Deriving the likelihood ratio for RGB images

This section provides full derivation of likelihood ratio (3) for RGB images.
Deriving the likelihood ratio for grayscale images is similar but won't be
covered here. Instead, intersted reader is referred to [1].

As it was mentioned in the introduction, we assume that the image $I$ is
**partitioned** into regions

$$
I = \bigcup\limits_{i=1}^{N} R_{i} ,
$$

where each $R_i$, for $i \in \{1, \ldots, N \}$, represents a region.

For each region, we assume that pixel values (intensity) are constant, but
corrupted by zero mean Gaussian noise.

Let $R_1$ and $R_2$ be regions made of $n$ and $m$ pixel, respectively. We'll
test the following hypothesis:

- $H_0$: pixel values from regions $R_1$ and $R_2$ come from the same distribution
- $H_1$: pixel values from regions $R_1$ and $R_2$ come from different distributions

<!-- Our assumption for the null hypothesis is that regions $R_1$ and $R_2$ belong to
the the same object, while alternative hypothesis says that regions $R_1$ and
$R_2$ belong to different objects. -->

We can test the two hypothesis by comparing the ratio of their likelihoods

$$
    L = \frac{p(x^{(1)}, x^{(2)}, \ldots, x^{(n + m)} | H_1)}
            {p(x^{(1)}, x^{(2)}, \ldots, x^{(n + m)} | H_0)} ,
$$

with some predefined threshold.

The null hypothesis assumes that regions $R_1$ and $R_2$ belong to the same
object, i.e. that the pixels are sampled from the same distribution

$$
p(x; \mu_0, \Sigma_0) =
\frac{1}{\sqrt{(2\pi)^{\text{D}}|\Sigma_0|}}
e^{-D(x, \mu_0, \Sigma_0)} ,
$$

where $x, \mu_0 \in \mathbb{R}^3$ and $\Sigma_0 \in \mathbb{R}^{3 \times 3}$.

The likelihood under $H_0$ can be written as

$$
\begin{align}
p(x^{(1)}, \ldots, x^{(n + m)} | H_0)
\stackrel{\text{i.i.d.}}{=}& \prod_{i=1}^{n + m} p(x^{(i)}|H_0) \\
=& \prod_{i=1}^{n + m} \frac{1}{\sqrt{(2\pi)^3|\Sigma_0|}}
e^{-\frac{1}{2}(x^{(i)} - \mu_0)^\text{T}\Sigma_0^{-1}(x^{(i)} - \mu_0)} \\
=& \left[(2\pi)^3|\Sigma_0|\right]^{-\frac{n + m}{2}}e^{-\frac{1}{2}\sum_{i=1}^{n + m}(x^{(i)} - \mu_0)^\text{T}\Sigma_0^{-1}(x^{(i)} - \mu_0)}
\end{align}
$$

We have to simplify the exponent:

$$
\begin{align}
\sum_{i=1}^{n + m}(x^{(i)} - \mu_0)^\text{T}\Sigma_0^{-1}(x^{(i)} - \mu_0)
&= \sum_{i=1}^{n + m}\text{Tr}\left[(x^{(i)} - \mu_0)^\text{T}\Sigma_0^{-1}(x^{(i)} - \mu_0)\right] \\
&= \sum_{i=1}^{n + m}\text{Tr}\left[(x^{(i)} - \mu_0)(x^{(i)} - \mu_0)^\text{T}\Sigma_0^{-1}\right] \\
&= \text{Tr}\left[\sum_{i=1}^{n + m}(x^{(i)} - \mu_0)(x^{(i)} - \mu_0)^\text{T}\Sigma_0^{-1}\right] \\
&= \text{Tr}\left\{\left[\sum_{i=1}^{n + m}(x^{(i)} - \mu_0)(x^{(i)} - \mu_0)^\text{T}\right]\Sigma_0^{-1}\right\} \\
\end{align}
$$

Expression in the brackets is un-normalized empirical
estimate of the covariance matrix. We can write

$$
(n + m)\text{Tr}\left[\left(\smash{\underbrace{\frac{1}{n + m}\sum_{i=1}^{n + m}(x^{(i)} - \mu_0)(x^{(i)} - \mu_0)^\text{T}}_{\hat{\Sigma}_0}}\right)\Sigma_0^{-1}\right]
$$

<br />

which yields

$$
\sum_{i=1}^{n + m}(x^{(i)} - \mu_0)^\text{T}\Sigma_0^{-1}(x^{(i)} - \mu_0) = (n + m)\text{Tr}\left[\hat{\Sigma}_0\Sigma_0^{-1}\right].
$$

Since $\hat{\Sigma}_0$ is an empirical estimate of $\Sigma_0$ and under the
assumption that $H_0$ is true, we can write $\hat{\Sigma}_0\Sigma^{-1}_0 = \text{I}_3$,
which yields

$$
\tag{4}
\sum_{i=1}^{n + m}(x^{(i)} - \mu_0)^\text{T}\Sigma_0^{-1}(x^{(i)} - \mu_0) = 3(n + m).
$$

Taking into account (4), we can write the final form of
the likelihood $p(x^{(1)}, \ldots, x^{(n + m)} | H_0)$ as

$$
\tag{5}
p(x^{(1)}, \ldots, x^{(n + m)} | H_0) = \left[(2\pi)^3|\hat{\Sigma}_0|\right]^{-\frac{n + m}{2}}e^{-\frac{3}{2}(n + m)},
$$

where we replaced the unknown covariance matrix $\Sigma_0$
with its empirical estimate $\hat{\Sigma}_0$, and $|\hat{\Sigma}_0$| denotes the determinant of $\hat{\Sigma}_0$.

Similar remarks can be made for likelihood under the
alternative hypothesis:

$$
\begin{align}
p(x^{(1)}, \ldots, x^{(n + m)} | H_1)
=& \prod_{i=1}^{n} p(x^{(i)}|H_1) \prod_{j=1}^{m} p(x^{(j)}|H_1) \\
\end{align}.
$$

The final expression for the likelihood under alternative hypothesis is

$$
\tag{6}
p(x^{(1)}, \ldots, x^{(n + m)} | H_1) = \left[(2\pi)^3|\hat{\Sigma}_1|\right]^{-\frac{n}{2}} e^{-\frac{3}{2}n}
\left[(2\pi)^3|\hat{\Sigma}_2|\right]^{-\frac{m}{2}} e^{-\frac{3}{2}m},
$$

where $\hat{\Sigma}_1$ and $\hat{\Sigma}_2$ are empirical estimates of $\Sigma_1$ and $\Sigma_2$, respectively,
for regions $R_1$ and $R_2$, under $H_1$.

Finally, we obtain the likelihood ratio from (5) and (6)

$$
L = \frac{|\hat{\Sigma}_0|^{\frac{n + m}{2}}}{|\hat{\Sigma}_1|^{\frac{n}{2}}|\hat{\Sigma}_2|^{\frac{m}{2}}}.
$$

## References

[1] Jain, Ramesh C., Kasturi, Rangachar and Schunck, Brian G.. Machine vision.. : McGraw-Hill, 1995.  
[2] Wasserman, Larry. *All of statistics : a concise course in statistical inference*. New York: Springer, 2010.