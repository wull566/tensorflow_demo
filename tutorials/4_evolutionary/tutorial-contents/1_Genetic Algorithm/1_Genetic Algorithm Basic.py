"""
遗传算法


"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 30            # DNA长度 DNA length
POP_SIZE = 100           # 种群大小 population size
CROSS_RATE = 0.8         # 交叉配对率  mating probability (DNA crossover)
MUTATION_RATE = 0.003    # 变异强度  mutation probability
N_GENERATIONS = 200      # 计算多少代
X_BOUND = [0, 10]         # x 轴范围 x upper and lower bounds


# 定义函数， 获取y最大高度
def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function


# 计算环境适应度
# find non-zero fitness for selection
def get_fitness(pred): return pred + 1e-3 - np.min(pred)
# def get_fitness(pred): return pred


# DNA 定义转换， 将二进制DNA转换为十进制，并将其规范化为一个范围（0,5）
# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop):
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


# 自然选择，适者生存
# pop 所有数据， fitness适应高的数据, 值越高，取到的概率越大
# np.random.choice (a=5, size=3, replace=True, p=[0.1,0.8])
# 从a中以概率p 可不一致 随机选择size=3个，replace代表抽样后是否放回去
def select(pop, fitness):    # nature selection wrt pop's fitness
    p = fitness/fitness.sum();
    # print(p * 10000)
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=p)
    # 概率越大，被选到的次数越多
    return pop[idx]


# 父母DNA交叉配对
def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE: # 80% 概率会配对
        i_ = np.random.randint(0, POP_SIZE, size=1)        # 随机选择一个配偶                        # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]    # 随机修改一部分DNA为配偶DNA                         # mating and produce one child
    return parent


# 基因变异
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:    # 0.003概率变异
            child[point] = 1 if child[point] == 0 else 0    # 变异修改基因中某个值
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

'''
pop = [
 [0 1 1 0 1 1 0 0 0 0]
 [1 1 0 1 1 1 1 0 1 1]
 [1 0 0 0 1 0 0 1 1 1]
 ...
]
'''

plt.ion()       # something about plotting
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))   # 画函数线

for _ in range(N_GENERATIONS):
    F_values = F(translateDNA(pop))    # compute function value by extracting DNA

    # something about plotting
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

    # GA part (evolution)
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child

plt.ioff()
plt.show()
