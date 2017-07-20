* 加载CSV数据集并去除空数据
```
challenger_data = np.genfromtxt('data/challenger_data.csv', skip_header=1, usecols=[1, 2], 
                                missing_values='NA', delimiter=',')
# drop the NA values
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]
# plot it, as a function of temperature (the first column)
print("Temp (F), O-Ring failure?")
print(challenger_data)
```

* 绘制散点图
```
plt.scatter(challenger_data[:, 0], challenger_data[:, 1], s=75, color="k",
            alpha=0.5)
plt.yticks([0, 1])
plt.ylabel("Damage Incident?")
plt.xlabel("Outside temperature (Fahrenheit)")
plt.title("Defects of the Space Shuttle O-Rings vs temperature");
```
![](http://ww2.sinaimg.cn/large/006HJ39wgy1fhqqfc2wscj31900eedh9.jpg)

* 正太分布
```
alpha = pm.Normal('alpha', 0, 0.0001, value=0)
```

* deterministic变量
```
@pm.deterministic
def p(t=temperature, a=alpha, b=beta):
    return 1.0 / (1 + np.exp(b * t + a))
```

* 伯努力分布
```
accident = pm.Bernoulli('bernoulli_obs', p, value=D, observed=True)
```

* 模型适配与训练
```
model = pm.Model([alpha, beta, accident])
map_ = pm.MAP(model)
map_.fit()
mcmc = pm.MCMC(model)
mcmc.sample(120000, 100000, 2)
```

* 模型抽样
```
alpha_samples = mcmc.trace('alpha')[:, None]
beta_samples = mcmc.trace('beta')[:, None]
```

* 绘制直方图
```
figsize(12.5, 6)

# histogram of the samples:
plt.subplot(211)
plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\beta$", color="#7A68A6", normed=True)
plt.legend()

plt.subplot(212)
plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\alpha$", color="#A60628", normed=True)
plt.legend();
```
![](http://ww3.sinaimg.cn/large/006HJ39wgy1fhqqjadx76j318q0kygni.jpg
)

* 将后验概率作图
```
t = np.linspace(temperature.min() - 5, temperature.max() + 5, 50)[:, None]
p_t = logistic(t.T, beta_samples, alpha_samples)
mean_prob_t = p_t.mean(axis=0)
figsize(12.5, 4)

plt.plot(t, mean_prob_t, lw=3, label="average posterior \nprobability \
of defect")
plt.plot(t, p_t[0, :], ls="--", label="realization from posterior")
plt.plot(t, p_t[-100, :], ls="--", label="realization from posterior")
plt.plot(t, p_t[-1000, :], ls="--", label="realization from posterior")
plt.scatter(temperature, D, color="k", s=50, alpha=0.5)
plt.title("Posterior expected value of probability of defect; \
plus realizations")
plt.legend(loc="lower left")
plt.ylim(-0.1, 1.1)
plt.xlim(t.min(), t.max())
plt.ylabel("probability")
plt.xlabel("temperature");
```
![](http://ww2.sinaimg.cn/large/006HJ39wgy1fhqqm4y9rfj317o0fiaf4.jpg)

* 0.95置信区间
```
from scipy.stats.mstats import mquantiles
qs = mquantiles(p_t, [0.025, 0.975], axis=0)

plt.fill_between(t[:, 0], *qs, alpha=0.7,
                 color="#7A68A6")

plt.plot(t[:, 0], qs[0], label="95% CI", color="#7A68A6", alpha=0.7)

plt.plot(t, mean_prob_t, lw=1, ls="--", color="k",
         label="average posterior \nprobability of defect")

plt.xlim(t.min(), t.max())
plt.ylim(-0.02, 1.02)
plt.legend(loc="lower left")
plt.scatter(temperature, D, color="k", s=50, alpha=0.5)
plt.xlabel("temp, $t$")

plt.ylabel("probability estimate")
plt.title("Posterior probability estimates given temp. $t$");
```
![](http://ww2.sinaimg.cn/large/006HJ39wgy1fhqqnj2el4j317m0fqq6y.jpg)

* 仿真数据
```
simulated = pm.Bernoulli('bernoulli_sim', p)
N = 10000

mcmc = pm.MCMC([simulated, alpha, beta, accident])
mcmc.sample(N)

figsize(12.5, 5)
simulations = mcmc.trace('bernoulli_sim')[:]
simulations.shape

for i in range(4):
    ax = plt.subplot(4, 1, i + 1)
    plt.scatter(temperature, simulations[1000 * i, :], color="k", s=50, alpha=0.6)
```
![](http://ww2.sinaimg.cn/large/006HJ39wgy1fhqqp9abbyj315w0h0gmx.jpg)

```
# 后验概率
posterior_probability = simulations.mean(axis=0)
print("posterior prob of defect | realized defect ")
for i in range(len(D)):
    print("%.2f                     |   %d" % (posterior_probability[i], D[i]))
```

```
# 风骚排序小技巧
ix = np.argsort(posterior_probability)
print("posterior prob of defect | realized defect ")
for i in range(len(D)):
    print("%.2f                     |   %d" % (posterior_probability[ix[i]], D[ix[i]]))
```