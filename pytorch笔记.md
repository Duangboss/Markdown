[toc]

***

# Ch3 Pytorch 入门

## Tensor

### 基本操作

- Tensor 是一种结构化的数据类型，可以理解为一个多维数组，是 PyTorch 中重要的数据结构。

- Tensor 和 Numpy 的 ndarrays 类似，但 Tensor 可以使用 GPU 进行加速。Tensor 的使用和 Numpy 的接口十分相似。

- Tensor 新建

  | 函数                    | 功能                      |
  | ----------------------- | ------------------------- |
  | Tensor(sizes)           | 基础构造函数              |
  | tensor(data,)           | 类似 np.array 的构造函数    |
  | ones(sizes)             | 全 1Tensor                 |
  | zeros(sizes)            | 全 0Tensor                 |
  | eye(sizes)              | 对角线为 1，其他为 0        |
  | arange(s, e, step)        | 从 s 到 e，步长为 step        |
  | linspace(s, e, steps)     | 从 s 到 e，均匀切分成 steps 份 |
  | rand/randn(sizes)       | 均匀/标准分布             |
  | normal(mean, std, sizes) | 正态分布/均匀分布         |
  | randperm(m)             | 随机排列 0-m               |

- `tensor.size() <=> tensor.shape` 查看 tensor 形状

- 在 PyTorch 中，许多张量操作函数都有两个版本：tensor.func 和 tensor.func_。前者返回一个新的张量，原始张量保持不变，后者对原始张量进行修改。

- 修改形状

  - `a.view() <=> a.resize()` 输出新 tensor
  - `a.resize_()` 改变原 tensor

- 索引方式

  - tensor 的索引方式与 ndarray 基本相似

  - 索引切片产生的结果与原 tensor 共享内存

  - `x[0, ...]` 和 `x[0, :]` 等价

  - tensor 索引函数

    - `index_select(input, dim, tensor_index)`：在指定维度 dim 上选取，比如选取某些行、某些列

    - `masked_select(input, bool_tensor)`：输出张量中满足掩码为 True 的元素

    - `non_zero(input)`：非 0 元素的下标

    - `gather(input, dim, tensor_index)`：根据 index，在 dim 维度上选取数据，输出的 size 与 index 一样

      ```python
      out[i][j] = input[index[i][j]][j]  # dim = 0
      out[i][j] = input[i][index[i][j]]  # dim = 1
      ```

    - `a.scatter_(1, index, b)` scatter 是 gather 的逆操作，根据索引张量在指定维度上将另一个张量的值 b 散布到当前张量 a 中。

  - `tensor[...,None,...] <=> tensor.unsqueeze(i)` 会在原张量的对应位置新建一个维度

  - `arr[idx]` 的返回值会是一个值，`ts[idx]` 返回的是一个 0 维度的 tensor，一般称为 scalar。使用 `scalar.item()` 取出值

- Tensor 拷贝

  - 深拷贝：`t.tensor(ts)` 和 `tensor.clone(ts)` 会进行数据拷贝且不共享内存。
  - 浅拷贝：使用 `torch.from_numpy(arr)`、`t.Tensor(arr)`、`ts.detach()` 或 `a = ts` 来新建一个 tensor 将共享内存。

- Tensor 类型

  - Tensor 中每个类别都有对应的 CPU 和 GPU 版本（**？？？除 HalfTensor**）

    | Data type                | dtype                             | CPU tensor           | GPU tensor                |
    | ------------------------ | --------------------------------- | -------------------- | ------------------------- |
    | 32-bit floating point    | `torch.float32` or `torch.float`  | `torch.FloatTensor`  | `torch.cuda.FloatTensor`  |
    | 64-bit floating point    | `torch.float64` or `torch.double` | `torch.DoubleTensor` | `torch.cuda.DoubleTensor` |
    | 16-bit floating point    | `torch.float16` or `torch.half`   | `torch.HalfTensor`   | `torch.cuda.HalfTensor`   |
    | 8-bit integer (unsigned) | `torch.uint8`                     | `torch.ByteTensor`   | `torch.cuda.ByteTensor`   |
    | 8-bit integer (signed)   | `torch.int8`                      | `torch.CharTensor`   | `torch.cuda.CharTensor`   |
    | 16-bit integer (signed)  | `torch.int16` or `torch.short`    | `torch.ShortTensor`  | `torch.cuda.ShortTensor`  |
    | 32-bit integer (signed)  | `torch.int32` or `torch.int`      | `torch.IntTensor`    | `torch.cuda.IntTensor`    |
    | 64-bit integer (signed)  | `torch.int64` or `torch.long`     | `torch.LongTensor`   | `torch.cuda.LongTensor`   |

  - 默认的 tensor 是 FloatTensor，可通过 `t.set_default_tensor_type` 来修改默认 tensor 类型

  - 使用 `xx.to(device) \ xx.cuda \ xx.cpu` 转换 CPU 和 GPU tensor

  - 类型转换

    - `tensor.type(new_type)`，不直接改变原张量，new_type 可以是 dtype、CPU tensor 和 GPU tensor
    - `tensor.to(dtype)`，不直接改变原张量，只能指定 dtype

  - `torch.*_like(tensor)` 和 `tensor.new_*()` 系列方法用于创建与现有张量具有相同形状或类型的新张量，*对应 ones、zeros、empty 等各种新建函数。

- 逐元素操作

  - 这部分操作会对 tensor 的每一个元素进行操作，输出形状与输入一直

    | 函数                            | 功能                                                       |
    | ------------------------------- | ---------------------------------------------------------- |
    | abs/sqrt/div/exp/fmod/log/pow.. | 绝对值/平方根/除法/指数/求余/求幂..                        |
    | cos/sin/asin/atan2/cosh..       | 相关三角函数                                               |
    | ceil/round/floor/trunc          | 上取整/四舍五入/下取整/只保留整数部分                      |
    | clamp(input, min, max)          | 将张量中的所有元素限制在 [min, max]，超过的部分替换为 min/max |
    | sigmod/tanh..                   | 激活函数                                                   |

- 归并操作

  - 对张量沿着某一维度或整个 tensor 进行指定操作

    | 函数                 | 功能                |
    | -------------------- | ------------------- |
    | mean/sum/median/mode | 均值/和/中位数/众数 |
    | norm/dist            | 范数/距离           |
    | std/var              | 标准差/方差         |
    | cumsum/cumprod       | 累加/累乘           |

  - 以上大多数函数都有一个 **dim** 参数，用来指定这些操作是在哪个维度上执行的。

    - 假设输入的形状是(m, n, k)
      - 如果指定 dim = 0，输出的形状就是(1, n, k)或者(n, k)
      - 如果指定 dim = 1，输出的形状就是(m, 1, k)或者(m, k)
      - 如果指定 dim = 2，输出的形状就是(m, n, 1)或者(m, n)
      - size 中是否有 "1"，取决于参数 `keepdim`，`keepdim=True` 会保留维度 `1`

- 比较函数

  |       函数        |                 功能                  |
  | :---------------: | :-----------------------------------: |
  | gt/lt/ge/le/eq/ne | 大于/小于/大于等于/小于等于/等于/不等 |
  |       topk        |              最大的 k 个数              |
  |       sort        |                 排序                  |
  |      max/min      |       比较两个 tensor 最大最小值        |

  - max/min 这两个函数较为特殊，以 max 举例：
    - t.max(tensor)：返回 tensor 中最大的数
    - t.max(tensor, dim)：指定维上最大的数，返回 tensor 和下标
    - t.max(tensor1, tensor2): 比较两个 tensor 相比较大的元素

- 线性代数

  | 函数                             | 功能                              |
  | -------------------------------- | --------------------------------- |
  | trace                            | 对角线元素之和(矩阵的迹)          |
  | diag                             | 对角线元素                        |
  | triu/tril                        | 矩阵的上三角/下三角，可指定偏移量 |
  | mm/bmm                           | 矩阵乘法，batch 的矩阵乘法         |
  | addmm/addbmm/addmv/addr/badbmm.. | 矩阵运算                          |
  | t                                | 转置                              |
  | dot/cross                        | 内积/外积                         |
  | inverse                          | 求逆矩阵                          |
  | svd                              | 奇异值分解                        |

### Tensor 和 Numpy

- Tensor 和 Numpy 数组之间具有很高的相似性，彼此之间的互操作也非常简单高效。
- 当遇到 Tensor 不支持的操作时，可先转成 Numpy 数组，处理后再转回 tensor，其转换开销很小

### 内部结构

- tensor 分为头信息区(Tensor)和存储区(Storage)
- 信息区主要保存着 tensor 的形状（size）、步长（stride）、数据类型（type）等信息；
- 存储区中的数据保存成连续数组，主要内存占用则取决于 tensor 中元素的数目。
  ![alt text](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/tensor_data_structure.png)

### 持久化

- 使用 t.save(a, filepath)保存

- 使用 t.load()加载

  ```Python
  b = t.load('a.pth')
  
  c = t.load('a.pth', map_location=lambda storage, loc: storage) # 加载为c, 存储于CPU
  
  d = t.load('a.pth', map_location={'cuda:1':'cuda:0'})# 加载为d, 存储于GPU0上
  ```

### 向量化

- 向量化计算是一种特殊的并行计算方式，可极大提高科学运算的效率.
- 相对于一般程序在同一时间只执行一个操作的方式，它可在同一时间执行多个操作，通常是对不同的数据执行同样的一个或一批指令，或者说把指令应用于一个数组/向量上。
- Python 语言中很多操作十分低效，尤其是 `for` 循环，在科学计算程序中应当极力避免使用 Python 原生的 `for循环`。

## autograd: 自动微分

### 手动计算梯度

- 将随机噪声添加到线性函数 y = x*2+3 中，模拟真实数据

- 损失函数 mse：$loss=\sum_{i}^{N}{\frac{1}{2}(y_i-(wx_i+b))^2}$

- 链式法则计算梯度：

  - $\partial loss = 1$
  - $\frac{\partial loss}{\partial y_{pred}}=y_{pred}-y_{true} => \partial y_{pred}=\partial loss * (y_{pred}-y_{true})$
  - $\frac{\partial loss}{\partial w}=\frac{\partial loss}{\partial y_{pred}}\cdot\frac{\partial y_{pred}}{\partial w}=x^t*(y_{pred} - y_{true})$
  - $\frac{\partial\text{loss}}{\partial b}=\frac{\partial\text{loss}}{\partial y_{\text{pred}}}\cdot\frac{\partial y_{\text{pred}}}{\partial b}=(y_{pred} - y_{true})$

- 对应代码

  ```Python
  dloss = 1
  dy_pred = dloss * (y_pred - y)
  dw = x.t().mm(dy_pred)
  db = dy_pred.sum()
  ```

### Autograd

- Autograd 是 PyTorch 中的自动微分引擎，它能够根据输入和前向传播过程自动构建计算图，并执行反向传播。
- 在神经网络训练中，我们需要的是对网络中的参数进行梯度计算，然后根据这些梯度更新参数，这一过程被称为反向传播。
- Autograd 的工作原理是定义一个计算图，图中的节点是 tensor，边是从输入 tensor 计算得到输出 tensor 的函数，计算图用于描述计算的过程。
- 当进行前向传播计算时，autograd 同时完成计算图的构建；当进行反向传播计算时，autograd 则根据计算图，利用链式法则进行梯度的计算。

### 

- require_grad

  - require_grad 用于指示张量是否需要梯度计算，设置为 True 时，PyTorch 会跟踪对该张量的所有操作，并自动计算梯度，以便进行反向传播和优化算法的更新。

    ```Python
    >> x = t.tensor(2, 2, requires_grad=True)
    >> # 等价于
    >> x = t.tensor(2,2)
    >> x.requires_grad = True
    ```

  - 变量的 `requires_grad` 属性默认为 False，如果某一个节点 requires_grad 被设置为 True，那么所有依赖它的节点 `requires_grad` 都是 True。

- Variable

  - Variable 是 autograd 中的核心数据结构，在 Pytorch 中已经被完全整合进了 Tensor 类中。可以认为需要求导(requires_grad)的 tensor 即 Variable。
  - 包含属性
    - data：保存 variable 所包含的 tensor
    - grad：保存 data 对应的梯度
    - grad_fn：指向一个 Function，记录 variable 的操作历史

- 调用 backward 方法，反向传播计算梯度

### 计算图

- 计算图是一种特殊的有向无环图（DAG），用于记录算子与变量之间的关系。
- 计算图的特点
  - autograd 根据用户对 variable 的操作构建其计算图。对变量的操作抽象为 `Function`。
  - 不是任何 Function 的输出，由用户创建的节点称为叶子节点，叶子节点的 `grad_fn` 为 None。叶子节点中需要求导的 variable，具有 `AccumulateGrad` 标识，因其梯度是累加的。
  - 如果某一个节点 requires_grad 被设置为 True，那么所有依赖它的节点 `requires_grad` 都为 True。
  - volatile 属性已被取消
  - 多次反向传播时，梯度是累加的。反向传播的中间缓存会被清空，为进行多次反向传播需指定 `retain_graph` = True 来保存这些缓存。
  - 非叶子节点的梯度计算完之后即被清空，可以使用 `autograd.grad` 或 `hook` 技术获取非叶子节点的值。
  - 直接修改 variable.data 的操作无法利用 autograd 进行反向传播，应避免
  - 反向传播函数 `backward` 的参数 `grad_variables` 可以看成链式求导的中间结果，如果是标量，可以省略，默认为 1
  - PyTorch 采用动态图设计，可以很方便地查看中间层的输出，动态的设计计算图结构。
- 一般用矩形表示算子，椭圆形表示变量.
- 例：
  - 表达式 $z = wx + b$ 可分解为 $y = wx$ 和 $z = y + b$，则其计算图如下图所示。
    ![alt text](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/com_graph.png)
  - 利用链式法求得各个叶子节点的梯度的流程如下
    $${\partial z \over \partial b} = 1,\space {\partial z \over \partial y} = 1\\
    {\partial y \over \partial w }= x,{\partial y \over \partial x}= w\\
    {\partial z \over \partial x}= {\partial z \over \partial y} {\partial y \over \partial x}= 1 * w\\
    {\partial z \over \partial w}= {\partial z \over \partial y} {\partial y \over \partial w}= 1 * x\\$$
  - 在反向传播过程中，autograd 沿着计算图从当前变量（根节点 $\textbf{z}$​）溯源，利用链式求导法则计算所有叶子节点的梯度。每一个前向传播操作的函数都有与之对应的反向传播函数用来计算输入的各个 variable 的梯度，而有了计算图，上述链式求导即可利用计算图的反向传播自动完成。
    ![alt text](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/com_graph_backward.png)

### 扩展 autograd

- 当需要定义 Function 时需做以下处理

  - 自定义的 Function 需要继承 autograd.Function，没有构造函数 `__init__`，需要重写 forward 和 backward 函数方法
  - backward 函数的输出和 forward 函数的输入一一对应，backward 函数的输入和 forward 函数的输出一一对应
  - backward 函数的 grad_output 参数即 t.autograd.backward 中的 `grad_variables`
  - 如果某一个输入不需要求导，直接返回 None，
  - 反向传播可能需要利用前向传播的某些中间结果，需要进行保存，否则前向传播结束后这些对象即被释放
  - Function 的使用利用 `Function.apply(variable)`

- 例：

  ```Python
  class MultiplyAdd(Function):
      @staticmethod
      def forward(ctx, w, x, b):                              
          ctx.save_for_backward(w,x)
          output = w * x + b
          return output
          
      @staticmethod
      def backward(ctx, grad_output):                         
          w,x = ctx.saved_tensors
          grad_w = grad_output * x
          grad_x = grad_output * w
          grad_b = grad_output * 1
          return grad_w, grad_x, grad_b 
  ```

## 使用流程

- 数据加载与预处理
- 定义网络
- 定义损失函数和优化器
- 训练网络
  - 输入数据
  - 前向传播+反向传播
  - 更新参数
- 测试网络

***

# Ch4 神经网络工具箱 nn

*   `torch.nn` 是 Pytorch 中专门为神经网络设计的模块化接口，用来定义和运行神经网络。

*   `nn.Module` 是 PyTorch 中构建神经网络模型的基础类。`nn.Module` 提供了一整套方法和属性，用于定义和管理神经网络的层、参数、前向传播和反向传播等操作。所有自定义的神经网络模型都应该继承自 `nn.Module` 类。其主要功能为：
    *   定义模型结构：通过在 `__init__` 方法中定义模型的层。
    *   前向传播：通过在 `forward` 方法中定义前向传播的计算过程。
    *   参数管理：通过内置的方法管理模型中的参数。
    *   模块嵌套：允许将多个子模块嵌套在一起，形成复杂的网络结构。
    
*   只要在 `Module` 的子类中定义了 `forward` 函数，`backward` 函数就会自动被实现(利用 `autograd`)。

*   在 `forward` 函数中可使用任何 tensor 支持的函数，还可以使用 if、for 循环、print、log 等 Python 语法。

*   网络的可学习参数通过 `net.parameters()` 返回，`net.named_parameters` 可同时返回可学习的参数及名称。

*   nn 中实现了神经网络中绝大多数的 layer，这些 layer 都继承于 `nn.Module`，封装了可学习参数

    *   如要自定义 layer，则该层中的参数需要在 `__init__()` 方法中定义

*   `nn.Module` 的输入输出都是 Tensor。只支持 `mini-batches`，不支持一次只输入一个样本，即一次必须是一个多个样本的批次（batch）。

*   如果只想输入一个样本，则需要使用 `input.unsqueeze(0)` 或 `input[None]` 将 batch\_size 设为 １

    *   `tensor.unsqueeze(idx)` 在 tensor 的第 idx 维位置添加一个新维度，如 idx = 0 则 n 维张量变为 1\*n 维张量。
    *   对应的 `tensor.squeeze()` 会移除张量中大小为 1 的维度
    *   batch\_size 对应 input 第一个维度的值

## 常用神经网络层

### 图像相关层

*   卷积层（Conv）

    *   卷积层广泛应用于图像处理和计算机视觉任务中。卷积层通过应用卷积核（滤波器）在输入数据上滑动，提取出特征图。
    ![alt text](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/same_padding_no_strides.gif)
    
*   `nn.Conv2d(in_channels=, out_channels=, kernel_size=, stride=, padding=)`
        *   `in_channels` 指定输入数据的通道数。（卷积核深度由输入通道数确认）
        *   `out_channels` 指定卷积层输出的特征图的通道数。(输出通道数由卷积核数确定)
        *   `kernel_size` 指定卷积核的大小，可以是整数或元组。
        *   `stride` 指定卷积核在输入图像上滑动的步。
        *   `padding` 指定在输入图像的边缘填充空白像素的数量。

*   池化层（Pool）

    *   池化层用于降低特征图的维度，同时保留重要特征，从而减少计算量和参数数量，并控制过拟合。

    *   `nn.MaxPool2d(kernel_size=2, stride=2, padding=0)`

        *   `kernel_size` 指定池化窗口的大小，可以是整数或元组。
        *   `stride` 指定池化窗口在输入图像上滑动的步。
        *   `padding` 指定在输入图像的边缘填充空白像素的数量。

    *   池化层的类型

        *   最大池化（Max Pooling）：选择池化窗口中的最大值。
        *   平均池化（Average Pooling）：计算池化窗口中的平均值。
        *   自适应池化（Adaptive Pooling）：允许指定输出特征图的尺寸，自动计算池化窗口的大小，以输出固定大小的特征图

*   常用层

    *   Linear：全连接层。

        *   全连接层用于执行线性变换 $y=Wx+b$
        *   `nn.Linear(in_features=128, out_features=64)`

    *   BatchNorm：批规范化层，分为 1D、2D 和 3D。

        *   批规范化层是用于加速深层神经网络训练并提高稳定性的方法。

        *   它通过对每一小批数据（mini-batch）的均值和方差进行标准化，减小不同小批之间的数据分布差异，从而使模型训练更加稳定。

            *   `nn.BatchNorm2d(num_features=64)`

    *   InstanceNorm：实例规范化层

        *   实例规范化层类似于批规范化层，但它在每个实例（样本）上单独计算均值和方差，常用于风格迁移。
        *   `nn.InstanceNorm2d(num_features=64)`

    *   Dropout：dropout 层，用来防止过拟合，同样分为 1D、2D 和 3D。

        *   Dropout 是一种防止神经网络过拟合的正则化技术。

        *   在训练过程中，Dropout 层会随机将一些神经元的输出置为零，并且在每个前向传播步骤中都会重新随机选择要置零的神经元。防止模型过度依赖某些特定的神经元，从而提高模型的泛化能力。

        *   `nn.Dropout(p=0.5)`

            *   `p` 指定每个元素被置为零的概率，默认为 0.5。

### 激活函数

*   激活函数为神经网络引入非线性特性，使神经网络能够学习和表示复杂的模式。

*   ReLU (Rectified Linear Unit) 激活函数

    *   $ReLU(x)=max(0,x)$

    *   特点：

        *   输出为输入的线性部分或零。
        *   计算简单，收敛速度快。

    *   优点：

        *   有效地减轻了梯度消失问题。
        *   更快的收敛速度。

    *   缺点：

        *   当输入为负时，梯度为零，可能导致“神经元死亡”现象。

*   Sigmoid 激活函数

    *   $σ(x)=\frac{1}{1+e^{−x}}$

    *   特点：

        *   输出范围在 (0, 1) 之间。
        *   对输入较大的正值或负值时，梯度会接近于零，可能导致梯度消失问题。
        *   通常用于输出层，特别是二分类问题。

    *   优点：

        *   平滑且连续，适用于概率输出。

    *   缺点：

        *   容易饱和，导致梯度消失。
        *   输出不为零均值，可能导致梯度更新缓慢。

*   Tanh (双曲正切) 激活函数

    *   $tanh(x)=\frac{e^{x}-e^{−x}}{e^{x}+e^{−x}}$

    *   特点：

        *   输出范围在 (-1, 1) 之间。
        *   相对于 Sigmoid 更适合隐藏层，因为它的输出均值为零。

    *   优点：

        *   输出范围更广，有助于收敛速度。
        *   相比 Sigmoid，有更快的梯度下降速度。

    *   缺点：

        *   仍然存在梯度消失问题。

*   Softmax 激活函数：

    *   $Softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$

    *   特点：

        *   将输入转换为概率分布，输出值在 (0, 1) 之间，且和为 1。
        *   通常用于多分类问题的输出层。

    *   优点：

        *   提供类别间的概率分布，有助于多分类问题。

    *   缺点：

        *   对大值敏感，可能导致数值不稳定。

### 前馈神经网络

*   前馈神经网络（FNN）是最简单的神经网络形式。在 FNN 中，信息只在一个方向上流动，从输入层，通过隐藏层，到输出层，没有任何循环或反馈连接。

*   对于此类网络如果每次都写复杂的 forward 函数会有些麻烦，在此就有两种简化方式，ModuleList 和 Sequential。

    * ModuleList

      * ModuleList 是一个容器，用于存储任意数量的 torch.nn.Module 子模块。可以包含不同类型的层，多个模块按照列表的方式组织起来，并且可能在运行时动态地添加或删除模块时。
    
      * ModuleList 不会自动处理输入数据的前向传播。
    
        ```python
        # 1
        modellist = nn.ModuleList([nn.Linear(3,4), nn.ReLU(), nn.Linear(4,2)])input = t.randn(1, 3)
        for model in modellist:
          input = model(input)
        # 2
        class MyModel(nn.Module):
          def __init__(self):
              super(MyModel, self).__init__()
              self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
          
          def forward(self, x):
              for layer in self.layers:
                  x = layer(x)
              return x
        ```
    
        
    
    *   Sequential
    
        *   与 ModuleList 相同，Sequential 将多个模块按顺序组织起来，不同的是他会自动处理前向传播过程，按照你添加模块的顺序依次传播数据。
        
        *   三种定义方式
        
            ```Python
            net1 = nn.Sequential()
            net1.add_module('conv', nn.Conv2d(3, 3, 3))
            net1.add_module('batchnorm', nn.BatchNorm2d(3))
            net1.add_module('activation_layer', nn.ReLU())
            
            net2 = nn.Sequential(
                      nn.Conv2d(3, 3, 3),
                      nn.BatchNorm2d(3),
                      nn.ReLU()
                    )
            
            from collections import OrderedDict
            net3= nn.Sequential(OrderedDict([
                      ('conv1', nn.Conv2d(3, 3, 3)),
                      ('bn1', nn.BatchNorm2d(3)),
                      ('relu1', nn.ReLU())
                    ]))
            print('net1:', net1)
            print('net2:', net2)
            print('net3:', net3)
            ```
        
        *   使用方式
        
            ```Python
            output = net1(input)
            
            output = net3.relu1(net2[1](net1.conv(input)))
            ```

### 循环神经网络

*   循环神经网络（RNN）是一类用于处理序列数据的神经网络。与 FNN 不同，RNN 允许信息在网络节点之间循环，能够处理任意长度的序列。
*   RNN 的核心思想是利用序列之前的信息来影响后续的输出。在 RNN 中，每个时间点上的单元都会接收到两部分输入：当前时间点的输入数据和上一个时间点单元的输出（也称为隐藏状态）。这种结构使得 RNN 能够在处理序列数据时考虑到前面的信息。
*   最常用的三种 RNN：RNN、LSTM 和 GRU。

    *   RNN
        ![alt text](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-3.png)
        $\begin{aligned}&h_{t}= Ux_{t}+Ws_{t-1}\\
        &s_{t}= f\big(h_{t}\big)\\
        &o_{t}= g\big(Vs_{t}\big)\end{aligned}$
    *   LSTM
        ![alt text](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/LSTM-Image.png)
        - LSTM 是一种特殊的循环神经网络，其保存着单元状态 C 和隐藏状态 h 两个神经元状态，使得 LSTM 能够学习和记忆长期依赖信息。该模型中包含三个重要的门结构：遗忘门 $f_t$、输入门 $i_t$ 和输出门
           $o_t$ $\begin{aligned} &f_t=\sigma(W_f\cdot[h_{t-1},x_t]+b_f) \\ &i_t=\sigma(W_i\cdot[h_{t-1},x_t]+b_i) \\ &o_t=\sigma(W_o\cdot[h_{t-1},x_t]+b_o) \\ &\tilde{C}_t=\tanh(W_C\cdot[h_{t-1},x_t]+b_C) \\ &C_t=f_t\cdot C_{t-1}+i_t\cdot\tilde{C}_t \\ &h_t=o_t\cdot\tanh(C_t) \end{aligned}$​
    *   GRU
        ![alt text](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-1-1721895042356-2.png)
        *   GRU 是 LSTM 的一种变体，将遗忘门和输入门合成了一个单一的更新门 $z_t$。同样还混合了细胞状态和隐藏状态。$\begin{aligned} &z_{t}=\sigma\left(W_{z}\cdot[h_{t-1},x_{t}]\right) \\ &r_{t}=\sigma\left(W_{r}\cdot[h_{t-1},x_{t}]\right) \\ &\tilde{h}_{t}=\tanh\left(W\cdot[r_{t}*h_{t-1},x_{t}]\right) \\ &h_{t}=(1-z_{t})*h_{t-1}+z_{t}*\tilde{h}_{t} \end{aligned}$
*   此外还有对应的三种 RNNCell，RNN 和 RNNCell 层的区别在于前者一次能够处理整个序列，而后者一次只处理序列中一个时间点的数据。
*   双向循环神经网络 BiRNN:

    *   双向循环神经网络 (BiRNN) 是对标准 RNN 的扩展，能够同时考虑序列数据的前向和后向信息。通过在每个时间步上进行前向和后向两个方向的计算，捕捉到序列中更多的上下文信息，从而提高模型的表现。
    *   一个双向 RNN 包含两个独立的 RNN，一个处理正向序列，另一个处理反向序列。

### 损失函数

*   损失函数（Loss Function）是在机器学习和深度学习中用于衡量模型预测和实际标签之间差异的函数。
*   常见损失函数

    *   回归问题

        *   MAE（L1 Loss）：nn.L1Loss

            *   在 $y=\hat{y}$ 处不可导
    *   MSE（L2 Loss）：nn.MSELoss
    
        *   MSE 对异常值更为敏感，会对误差较大的值基于更大的惩罚    ![alt text](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image.png)
    *   Huber Loss:
    
        *   $L_\delta(y,\hat{y})=\begin{cases}\frac12(y-\hat{y})^2&\mathrm{if}|y-\hat{y}|\leq\delta\\\delta|y-\hat{y}|-\frac12\delta^2&\mathrm{otherwise}\end{cases}$
    
            *   $\delta$ 是阈值，控制误差的界限。
    
        *   结合了 MAE 和 MSE，能够对异常值进行鲁棒处理，同时在误差较小时表现良好
    *   分类问题：
    
        *   交叉熵损失：
    
            *   $\mathrm{Loss}=-\frac1N\sum_{i=1}^N\sum_{c=1}^Cy_{i,c}\log(p_{i,c})$
    
        *   二元交叉熵损失（BCE）
    
            *   $\mathrm{Loss}=-\frac1N\sum_{i=1}^N\left(y_i\log(\hat{y}_i)+(1-y_i)\log(1-\hat{y}_i)\right)$
        *   主要用于二分类任务。
    
        *   负对数似然损失（NLL）
    
            *   $\mathrm{Loss}=-\sum_{i=1}^N\log p(y_i)$
        *   主要用于分类任务，通常与 LogSoftmax 一起使用。
    
        *   Hinge 损失
    
            *   $L(y,\hat y)=\max(0,1-y_i\cdot\hat y_i)$
        *   常用于二分类问题中，特别是 SVM 中。

## 优化器

*   优化器在神经网络训练中用于更新模型的参数以最小化损失函数，优化器的选择和调参对于训练神经网络模型的性能和收敛速度有着重要的影响。

*   在 PyTorch 中，`torch.optim` 模块提供了各种优化器的实现，常见的优化器包括：

    *   `torch.optim.SGD`：随机梯度下降优化器，通过计算每个参数的梯度并在参数空间中沿着负梯度方向更新参数来最小化损失。
    *   `torch.optim.Adam`：Adam 优化器，结合了动量法和自适应学习率的特性，通常在深度学习中表现良好。

*   使用步骤：

    ```Python
    optimizer = optim.SGD(net.parameters(), lr = 0.01) # 实例化优化器，指定要调整的参数和学习率
    optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
    
    output = net(input)
    loss = criterion(output, target # 计算损失
    
    loss.backward() # 反向传播
    
    optimizer.step() # 更新模型参数
    ```

*   为不同子网络设置不同学习率

    ```Python
    optimizer=optim.SGD([
                  {'params': net.features.parameters(), 'lr': 1e-5},
                  {'params': net.classifier.parameters(), 'lr': 1e-2}
              ])
    ```

## nn.functional（F）

*   在 Pytorch 中，nn 中的大部分 layer、损失函数、激活函数在 F（torch.nn.functional）中都有相应的函数
* nn 中的 layer、损失函数是以类（class）的形式定义的，它们是模块化的。使用时需要先实例化一个对象，再调用这个对象来计算损失。F 中的相应函数的形式定义的，使用时直接调用函数即可，无需实例化。

  ```Python
  # nn
  model = nn.Linear(3, 4)
  output1 = model(input)
  #F
  output2 = nn.functional.linear(input, model.weight, model.bias)
  
  # nn
  loss_fn = nn.CrossEntropyLoss()
  loss = loss_fn(output, target)
  # F
  loss = F.cross_entropy(output, target)
  ```
* 如果模型有可学习的参数，最好用 nn.Module，否则既可以使用 `F` 也可以使用 nn.Module，二者在性能上没有太大差异。如激活函数、池化等层。

*   另外虽然 dropout 操作也没有可学习操作，但建议还是使用 `nn.Dropout` 而不是 `F.dropout`，因为 dropout 在训练和测试两个阶段的行为有所差别，使用 `nn.Module` 对象能够通过 `model.eval` 操作加以区分。

## 初始化策略

* PyTorch 中 nn.Module 的模块参数都采取了较为合理的初始化策略，一般不用做多余处理，需要时可以用自定义初始化去代替系统的默认初始化。

* PyTorch 中 `nn.init` 模块就是专门为初始化而设计，如果某种初始化策略 `nn.init` 不提供，用户也可以自己直接初始化。

  ```python
  linear = nn.Linear(3, 4) nn.init.xavier\_normal_(linear.weight)
  ```

## nn.Module 深入分析

* nn.Module 基类的构造函数：

  ```Python
  def __init__(self):
      self._parameters = OrderedDict()
      self._modules = OrderedDict()
      self._buffers = OrderedDict()
      self._backward_hooks = OrderedDict()
      self._forward_hooks = OrderedDict()
      self.training = True
  ```

  - `_parameters`：字典，保存用户直接设置的 parameter，`self.param1 = ...` 会被检测到，在字典中加入一个 key 为'param1'，value 为对应 parameter 的 item。而 self.submodule = nn.Linear(3, 4)中的 parameter 则不会存于此。

  *   `_modules`：子 module，通过 `self.submodel = nn.Linear(3, 4)` 指定的子 module 会保存于此。
  *   `_buffers`：缓存。如 batchnorm 使用 momentum 机制，每次前向传播需用到上一次前向传播的结果。
  *   `_backward_hooks` 与 `_forward_hooks`：钩子技术，用来提取中间变量，类似 variable 的 hook。
  *   `training`：BatchNorm 与 Dropout 层在训练阶段和测试阶段中采取的策略不同，通过判断 training 值来决定前向传播策略。
  *   上述几个属性中，`_parameters`、`_modules` 和 `_buffers` 这三个字典中的键值，都可以通过 `self.key` 方式获得，效果等价于 `self._parameters['key']`.

*   `getattr` 和 `setattr`

    *   result = obj.name 会调用内置函数 `getattr(obj, 'name')`，如果该属性找不到，会调用自定义的属性访问方法 `obj.__getattr__('name')`
    *   obj.name = value 会调用内置函数 `setattr(obj, 'name', value)`，如果 obj 对象实现了 `__setattr__` 方法，`setattr` 会调用自定义的属性设置方法 `obj.__setattr__('name', value')`

*   保存模型

    * 所有的 Module 对象都具有 `state_dict()` 函数，返回当前 Module 所有的状态数据。将这些状态数据保存后，下次使用模型时即可利用 `model.load_state_dict()` 函数将状态加载进来。 
    
      ```python
      t.save(net.state_dict(), 'net.pth')
      net2 = Net() net2.load_state_dict(t.load('net.pth'))
      ```

# Ch5 PyTorch 常用工具模块

## 数据处理

### 数据加载

- 数据集 Dataset

  - Dataset 对象是一个数据集，只负责数据的抽象，可以使用 `ds[i]` 和 `ds.__getitem__(i)` 访问，返回形如(data, label)的数据。

  - 要创建一个自定义数据集，需要继承 `torch.utils.data.Dataset` 并实现以下方法：
    `__len__`: 返回数据集的大小。
    `__getitem__`: 支持索引操作，使得 ds [i] 可以获取第 i 个样本。

    ```Python
    class CustomDataset(Dataset):
        def __init__(self, data, labels):
          self.data = data
          self.labels = labels
    
        def __len__(self):
          return len(self.data)
    
        def __getitem__(self, idx):
          sample = self.data[idx]
          label = self.labels[idx]
          return sample, label
    ```

*   Dataloader

    *   `DataLoader` 结合了 `Dataset` 对象，提供了一种简便的方法来迭代数据集。`DataLoader` 可以自动地将数据分批（batch）加载，并支持多线程数据加载，这对处理大型数据集尤其有用。

    *   `DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False)`

        *   `dataset`：一个继承自 Dataset 的对象。

        *   `batch_size`：每个批次的数据量。

        *   `shuffle`：是否在每个 epoch 开始时打乱数据。

        *   `sampler`：样本抽样，后续会详细介绍

            *   指定 sampler 参数时，shuffle 参数将被忽略。

            *   RandomSampler：随机采样器，在每个 epoch 中随机打乱数据顺序。

                *   `DataLoader(dataset, sampler=RandomSampler(dataset), ...)`

            *   SequentialSampler：顺序采样器，按照数据集的顺序逐个采样。

                *   `DataLoader(dataset, sampler=SequentialSampler(dataset), ...)`

            *   SubsetRandomSampler：子集随机采样器，从数据集中随机采样一个子集。

                *   `DataLoader(dataset, sampler=SubsetRandomSampler(indices), batch_size=32)`

            *   WeightedRandomSampler：加权随机采样器，根据指定的权重进行采样，适用于类别不平衡的数据。

                *   `WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)`

                    *   `weights` 中包含每个样本的权重
                    *   `num_samples` 是采样的总数
                    *   `replacement` 控制是否允许重复采样

            *   继承 sampler 类来自定义 sampler

        *   `num_workers`：用于数据加载的子进程数。

        *   `collate_fn`：如何将多个样本合并成一个小批次的函数。（默认情况下，它将样本堆叠到一个小批次中）

        *   `pin_memory`：是否将数据保存在 `pin memory` 区，`pin memory` 中的数据转到 GPU 会快一些

        *   `drop_last`：如果设置为 True，则在数据集大小不能被批次大小整除时，删除最后一个不完整的批次。

    *   迭代方法

        *   `for idx, batch in enumerate(dataloader)`
        *   `for batch in dataloader`
        *   `dataiter=iter(dataloader) batch=next(dataiter)`

## 计算机视觉工具包：torchvision

*   torchvision 是 PyTorch 中专门针对计算机视觉任务设计的软件包，提供了许多用于处理图像和视觉数据的工具和实用功能。torchvision 主要包含以下功能：

*   数据集 `torchvision.datasets`：

    *   `datasets` 模块提供了常用的计算机视觉数据集，如 MNIST、CIFAR-10 等，方便用户加载和使用这些数据集进行训练和测试。

*   数据处理 `torchvision.transforms`：

    *   `transforms` 提供了一些常用的图像变换操作。这些变换可以应用于图像，以便为训练神经网络或执行其他计算机视觉任务做准备。

    *   对 PIL Image 的操作：

        *   `ToTensor()`：将 PIL 图像或 numpy.ndarray 转换为张量。

            *   将图像从 (H, W, C) 形状转换为 (C, H, W)，并将像素值归一化到\[0.0, 1.0]。

        *   `Normalize(mean, std)`：使用均值和标准差对张量图像进行归一化。

        *   `Resize(size)`：将输入图像调整为给定大小。

            *   `size` 参数可以是整数或元组 (height, width)。

        *   `CenterCrop(size)`：将图像的中心部分裁剪为给定大小。

        *   `RandomCrop(size)`：随机裁剪图像的一部分为给定大小。

        *   `RandomHorizontalFlip(p)`：以概率 `p` 水平翻转图像。

        *   `RandomRotation(degrees)`：随机选择一个角度从给定范围内旋转图像。

        *   `Pad(padding, fill)`：在图像边缘填充像素。

        *   `ColorJitter(brightness, contrast, saturation, hue)`：随机改变图像的亮度、对比度、饱和度和色调。

        *   `RandomAffine(degrees, translate, scale, shear)`：对图像应用随机仿射变换。

        *   `Grayscale(num_output_channels)`：将图像转换为灰度图像。

        *   `ToPILImage`：将 Tensor 转为 PIL Image 对象

    *   `Compose(transforms)`：将多个变换组合在一起。例如，您可以将多个变换串联起来。

    *   示例 transform = transforms.Compose(\[ transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean =\[0.485, 0.456, 0.406], std =\[0.229, 0.224, 0.225]) ]) # 定义 transform image = Image.open("path/to/your/image.jpg") # 加载图像 transformed\_image = transform(image) # 应用变换

*   模型 `torchvision.models`

    *   `models` 模块包含了一些经典的计算机视觉模型，如 ResNet、VGG、AlexNet 等，用户可以直接使用这些预训练好的模型进行图像分类、目标检测等任务。

*   工具 `torchvision.utils`：

    *   `utils` 模块提供了一些实用的函数和工具，如保存和加载图像、可视化模型预测结果等。

        *   `make_grid(imgs, nrow)`：将多张图片拼接成一个网格中；
        *   `save_image(img, filename)`: 将 Tensor 保存成图片。

## 可视化工具

### visdom

*   Visdom 是专门为 PyTorch 开发的一款可视化工具

*   Visdom 同时支持 PyTorch 的 tensor 和 Numpy 的 ndarray 两种数据结构，但不支持 Python 的 int、float 等类型。

*   Visdom 的两个重要概念：

    *   env：环境。不同环境的可视化结果相互隔离，互不影响，在使用时如果不指定 env，默认使用 `main`。不同用户、不同程序一般使用不同的 env。
    *   pane：窗格。窗格可用于可视化图像、数值或打印文本等，其可以拖动、缩放、保存和关闭。一个程序中可使用同一个 env 中的不同 pane，每个 pane 可视化或记录某一信息。

*   `python -m visdom.server` 命令启动 visdom

    *   或通过 `nohup python -m visdom.server &` 后台启动

*   连接客户端：`vis = visdom.Visdom(env=)`

*   使用 visdom 画图，数据的指定形式与 pyplot 类似，`vis.line(X=x, Y=y, )`，主要有以下特殊参数

    *   win：用于指定 pane 的名字，如果不指定，visdom 将自动分配一个新的 pane。

    *   opts：选项，接收一个字典，常见的 option 包括 `title`、`xlabel`、`ylabel`、`width` 等，主要用于设置 pane 的显示格。

    *   `update`：如果要在再次操作已有 win，则需要指定 update 参数（第一次创建 win 时不可以指定）。
        *   当 `update='append'` 时，为向已有线上（使用 name 参数标识）添加数据点。
        *   当 `update='new'` 时，为新建一条线
    
*   image 的画图功能可分为如下两类：

    *   `image` 接收一个二维或三维向量，$H\times W$ 或 $3 \times H\times W$，前者是黑白图像，后者是彩色图像。
    *   `images` 接收一个四维向量 $N\times C\times H\times W$，$C$ 可以是 1 或 3，分别代表黑白和彩色图像，将多张图片拼接在一起。`images` 也可以接收一个二维或三维的向量，此时它所实现的功能与 image 一致。

*   `vis.text` 用于可视化文本，支持所有的 html 标签，遵循 html 的语法标准。

## 使用 GPU 加速：cuda

*   在 PyTorch 中以下数据结构分为 CPU 和 GPU 两个版本：

    *   Tensor
    *   nn.Module（包括常用的 layer、loss function，以及容器 Sequential 等）

*   可以使用 `device = t.device("cuda:0" if t.cuda.is_available() else "cpu")` 判断是否存在 CUDA 设备并选择相应的设备

*   如果有多个 GPU，使用 torch.device(f'cuda:{i}') 来表示第 i 块 GPU（i 从 0 开始）。cuda: 0 和 cuda 是等价的。

*   使用 `xx.to(device) \ xx.cuda() \ xx.cpu()` 转移设备

    *   `tensor.cuda` 会返回一个新对象，这个新对象的数据已转移至 GPU，而之前的 tensor 还在原来的设备上。
    *   `module.cuda` 则会将所有的数据都迁移至 GPU，并返回自己。

*   使用 `tensor.device \ tensor.is_cuda` 查看张量所处设备

*   使用 `torch.set_default_tensor_type` 使程序默认使用 GPU

*   torch 网络在 CPU 或 GPU 上运行时，要保证其输入、网络、优化器、损失函数都在用一个设备上

*   torch 网络在 CPU 上训练时会默认将线程数设置为 CPU 核心数，可以使用 `t.set_num_threads(x)`

***

# Ch6 猫和狗二分类

## 项目介绍
Dogs vs. Cats 是 kaggle 网站举办的一个挑战赛 [https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition]，为二分类问题，竞赛中提供了训练集和测试集。训练集包含 25000 张图片，命名格式为 `<category>.<num>.jpg`, 如 `cat.10000.jpg`、`dog.100.jpg`，测试集包含 12500 张图片，命名为 `<num>.jpg`，如 `1000.jpg`。参赛者需根据训练集的图片训练模型，并在测试集上进行预测，输出它是狗的概率。最后提交的 csv 文件如下，第一列是图片的 `<num>`，第二列是图片为狗的概率。
```
id,label
10001,0.889
10002,0.01
...
```

## 项目构建
### 数据加载与预处理
- 从 [竞赛官网下载数据集](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
- 使用 `tv.datasets.ImageFolder` 或 `tv.datasets.DatasetFolder` 加载数据集
  - 两者都是读入给定 folder_path 中子目录中的图片
  - 两者会直接将图片读入内存，在访问到对应图片时才会读入
- 自定义数据集类
  ```Python
  class DogCat(Dataset):
      def __init__(self, folderpath, mode="train", transforms=None):
          self.mode = mode
  
          # 获取所有图片路径
          imgs = [folderpath + img for img in os.listdir(folderpath)]
          
          # 判断数据集类型，如果模式为测试集，则读入所有图片
          # 其他情况使用train_test_split将数据集分为训练集和验证集
          if self.mode != "test":
              train, valid = train_test_split(imgs, random_state=seed, test_size=0.3, )
              if self.mode=="train":
                  self.imgs = train
              else:
                  self.imgs = valid
  
          # 如果数据集类型为训练集则需要对图片进行随机缩放裁切和随机旋转进行数据增强
          if transforms is None:
              normalize = T.Normalize(mean = [0.485, 0.456, 0.406], 
                                      std = [0.229, 0.224, 0.225])
              if self.mode=="test": 
                  self.transforms = T.Compose([
                      T.Resize(224),
                      T.CenterCrop(224),
                      T.ToTensor(),
                      normalize
                  ]) 
              else:
                  self.transforms = T.Compose([
                      T.Resize(256),
                      T.RandomResizedCrop(224),
                      T.RandomHorizontalFlip(),
                      T.ToTensor(),
                      normalize
                  ])
  
          # x.split('/')[-1]从路径中分出文件名，split('.')[0]分出图片标签，训练集、验证集为类型，测试集为id
          self.label = [x.split('/')[-1].split('.')[0] for x in self.imgs]
      
      # 将文件读取放在`__getitem__`中，在通过dataloader加载数据利用多进程加速。
      # 一次性将所有图片都读进内存，不仅费时也会占用较大内存，而且不易进行数据增强等操作。
      def __getitem__(self, index):
          img_path = self.imgs[index]
          data = self.transforms(Image.open(img_path))
          
          return data, self.label[index]
      
      def __len__(self):
          return len(self.imgs)
  ```
- 在训练过程中使用使用 dataloader 迭代数据集，使用 tpdm 可视化迭代进度
### 模型选择
- SimpleCNN
  - SimpleCNN 通常指的是一种简单的卷积神经网络，一般由几层卷积层和池化层组成。它是入门级的 CNN 架构。
    ```Python
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.AvgPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 28 * 28, 512)
            self.fc2 = nn.Linear(512, 2)
    
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    ```
- AlexNet
  - AlexNet 是 2012 年由 Alex Krizhevsky 等人提出的一种深度卷积神经网络。它在 ImageNet 图像分类挑战中取得了巨大成功，被认为是现代深度学习的里程碑之一。
    ![alt text](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-1.png)
    
    - AlexNet 展示了通过使用多层 CNN，可以在大规模图像分类任务上取得显著的性能提升。它证明了深度模型能够学习到更复杂、抽象的特征表示，这对于提高模型的准确性和泛化能力至关重要。
    - AlexNet 大量使用了 ReLU 激活函数，相比于传统的 sigmoid 和 tanh 激活函数，ReLU 的计算效率更高，能够缓解梯度消失问题，加速训练过程。
    - AlexNet 引入了 Dropout 作为正则化手段，通过随机关闭一部分神经元，防止过拟合，从而提高模型的泛化性能。
  - 模型定义
    ```Python
    from torch import nn
    from .basic_module import BasicModule
    
    class AlexNet(BasicModule):
        def __init__(self, num_classes=2):
            super(AlexNet, self).__init__()
    
            self.model_name = 'alexnet'
    
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
    
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)
            return x
    ```
  
- SqueezeNet
  - SqueezeNet 是一种轻量级的 CNN 结构，由 Iandola 等人在 2016 年提出。它的目标是在减少模型参数数量的同时保持较高的准确率。SqueezeNet 的参数量比 AlexNet 减少了 50 倍，但在 ImageNet 上精度相似。它使用了一种叫做 "Fire 模块" 的结构来实现这一目标。
  - 主要思路
    - 使用 1 × 1 卷积核代替 3 × 3 卷积核，减少参数量；
    - 通过 squeeze layer 限制通道数量，减少参数量；
    - 减少池化层（下采样），并将池化操作延后，给卷积层带来更大的激活层，保留更多地信息，提高准确率；
    - 避免使用全连接层，使用全局平均池化代替全连接层;
  - Fire 模块
    ![alt text](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-2.png)
    
    - Squeeze 层：通常包含 1x1 的卷积核，用于减少输入特征图的通道数，从而压缩数据的维度，减少后续计算的参数数量。
    - Expand 层：随后使用 1x1 和 3x3 的卷积核来恢复或增加通道数，这些卷积操作在较少的输入通道上执行，从而大大减少了参数的数量。
    ```Python
    class Fire(nn.Module):
        def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
            super(Fire, self).__init__()
            self.in_channels = in_channels
            self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
            self.squeeze_activation = nn.ReLU(inplace=True)
            self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels,
                                    kernel_size=1)
            self.expand1x1_activation = nn.ReLU(inplace=True)
            self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels,
                                    kernel_size=3, padding=1)
            self.expand3x3_activation = nn.ReLU(inplace=True)
    
        def forward(self, x):
            x = self.squeeze_activation(self.squeeze(x))
            e1 = self.expand1x1_activation(self.expand1x1(x))
            e2 = self.expand3x3_activation(self.expand3x3(x))
            out = torch.cat([e1, e2], 1)
            return out
    ```
    
  - 网络定义
    ```Python
    class SqueezeNet(nn.Module):
        def __init__(self, version='1_0', num_classes=1000):
            super(SqueezeNet, self).__init__()
            self.num_classes = num_classes
            if version == '1_0':
                self.features = nn.Sequential(
                    nn.Conv2d(3, 96, kernel_size=7, stride=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(96, 16, 64, 64),
                    Fire(128, 16, 64, 64),
                    Fire(128, 32, 128, 128),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(256, 32, 128, 128),
                    Fire(256, 48, 192, 192),
                    Fire(384, 48, 192, 192),
                    Fire(384, 64, 256, 256),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(512, 64, 256, 256),
                )
    
            elif version == '1_1':
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(64, 16, 64, 64),
                    Fire(128, 16, 64, 64),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(128, 32, 128, 128),
                    Fire(256, 32, 128, 128),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(256, 48, 192, 192),
                    Fire(384, 48, 192, 192),
                    Fire(384, 64, 256, 256),
                    Fire(512, 64, 256, 256),
                )
    
            final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
    
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            out = torch.flatten(x, 1)
            return out
    ```
- ResNet34
  - ResNet（Residual Network）是由何恺明等人在 2015 年提出的一种深度卷积神经网络，主要通过引入残差连接（skip connections）来解决深层网络中的梯度消失问题。ResNet34 是 ResNet 的一个变种，包含 34 层深度。
  - 残差快
    ```Python
    class ResidualBlock(nn.Module):
        def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
            super(ResidualBlock, self).__init__()
            self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(outchannel))
            self.right = shortcut
    
        def forward(self, x):
            out = self.left(x)
            residual = x if self.right is None else self.right(x)
            out += residual
            return F.relu(out)
    ```
  - 残差网络
    ```Python
    class ResNet34(BasicModule):
        def __init__(self, num_classes=2):
            super(ResNet34, self).__init__()
            self.model_name = 'resnet34'
    
            # 前几层: 图像转换
            self.pre = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))
    
            # 重复的layer，分别有3，4，6，3个residual block
            self.layer1 = self._make_layer(64, 128, 3)
            self.layer2 = self._make_layer(128, 256, 4, stride=2)
            self.layer3 = self._make_layer(256, 512, 6, stride=2)
            self.layer4 = self._make_layer(512, 512, 3, stride=2)
    
            # 分类用的全连接
            self.fc = nn.Linear(512, num_classes)
    
        def _make_layer(self, inchannel, outchannel, block_num, stride=1):
            """
            构建layer,包含多个residual block
            """
            shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                nn.BatchNorm2d(outchannel))
    
            layers = []
            layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
    
            for i in range(1, block_num):
                layers.append(ResidualBlock(outchannel, outchannel))
            return nn.Sequential(*layers)
    
        def forward(self, x):
            x = self.pre(x)
    
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
    
            x = F.avg_pool2d(x, 7)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    ```
- torchvision 提供了许多预训练的计算机视觉模型，比如 ResNet、VGG、MobileNet 等等。使用 pretrained 参数来加载预训练的模型权重。
    ```Python
    class SqueezeNet(BasicModule):
        def __init__(self, num_classes=2):
            super(SqueezeNet, self).__init__()
            self.model_name = 'squeezenet'
            self.model = squeezenet1_1(pretrained=True)
            # 修改 原始的num_class: 预训练模型是1000分类
            # torchvision 中的预训练模型大多数都是在 ImageNet 数据集上训练的
            # ImageNet 数据集包含 1000 个类别。
            # 因此，预训练的模型默认情况下都是用于 1000 类分类任务的。
            self.model.num_classes = num_classes
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, num_classes, 1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(13, stride=1)
            )
    
        def forward(self,x):
            return self.model(x)
    
        # 在使用预训练模型的情况下，我们只需要训练自定义的分类器，CNN部分权重可以保持不变
        def get_optimizer(self, lr, weight_decay):
            return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay) 
    ```
  - 使用预训练模型的注意事项
    - 图像预处理
      - 输入图像格式通常需要进行以下预处理步骤：
        - 调整尺寸：将图像调整为模型所需的尺寸（如 224x224）。
        - 归一化：将图像像素值归一化到 [0, 1] 范围，并根据模型预训练时的均值和标准差进行标准化。
    - 输入要求
      - 查看模型的文档或源代码来获取输入图像的要求。以下是一些常见模型的输入要求
        - ResNet、VGG、DenseNet 等：通常要求输入图像为 3 个通道（RGB），尺寸为 224x224 像素。
        - Inception：输入尺寸通常为 299x299 像素。

- 模型训练
  - 用到的模块
    - torchnet.meter
      - torchnet 库提供了许多工具来简化使用 PyTorch 训练神经网络的过程。meter 模块它提供了多种度量方法来评估模型的性能和训练过程中的各种指标。
      - AverageValueMeter: 用于计算一组值的平均值，常用于计算损失函数的平均值。
        ```Python
        meter = meter.AverageValueMeter()
        meter.add(value)
        mean = meter.value()
        ```
      - ConfusionMeter: 这个类用于计算混淆矩阵，适用于分类任务。
        ```Python
        meter = meter.ConfusionMeter(num_classes)
        meter.add(predicted, target)
        cm_value = meter.value()
        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
        ```
        | 样本   | 判为狗  | 判为猫  |
        | ---- | ---- | ---- |
        | 实际是狗 | 35   | 15   |
        | 实际是猫 | 9    | 91   |

    - fire
      - Python Fire 是一个由 Google 开发的库，可以将 Python 对象（如类、函数或字典）转换为可以从终端运行的命令行工具，旨在通过命令行轻松调用 Python 程序。
      - 使用 fire 将函数转换为命令行工具
        ```python
        # greet.py
        import fire
        
        def greet(name, greeting='Hello'):
            return f"{greeting}, {name}!"
        
        if __name__ == '__main__':
            fire.Fire()
        ```
      - 然后在命令行中运行：
        ```
        python greet.py greet --name=World --greeting=Hi
        ```

***

# Ch7 生成对抗网络 GAN

## GAN 概述

- 基本结构
  ![image-20240722222628472](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240722222628472.png)
  - 生成器
    - 生成器的任务是从一个随机噪声（通常是高斯噪声）中生成逼真的数据样本。生成器是一个神经网络，它接受一个随机向量作为输入，并输出一个与真实数据分布相似的样本。
    - 目标是欺骗判别器，使其认为生成的数据是真实的。
  - 判别器
    - 判别器是另一个神经网络，用于区分真实数据和生成器生成的假数据。它的输入是一个数据样本，输出是一个标量值，表示输入数据是真实的概率。
    - 判别器的目标是最大化区分真实数据和生成数据的准确性。
  - 训练判别器时， 需要利用生成器生成的假图片和来自真实世界的真图片； 训练生成器时，只用噪声生成假图片 。 判别器用来评估生成的假图片的质量，促使生成器相应地调整参数 。这二者的目标相反，在训练过程中互相对抗，这也是它被称为生成对抗网络的原因 。
- 工作原理

  - GANs 通过生成器和判别器的对抗训练来实现目标。训练过程中，两者的目标是相反的：
    - 生成器试图最大限度地欺骗判别器，使其无法区分生成数据和真实数据。
    - 判别器则尽量提高区分真实数据和生成数据的能力。
  - GAN 可以形式化为一个极小极大（minimax）问题：
    ![在这里插入图片描述](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/3a082ec2790b47258a884a1281515f1a.png)
    - 其中，$D(x)$ 是判别器对真实数据 $x$ 的输出，$p_{data}(x)$ 是真实数据的分布；$G(z)$ 是生成器对随机噪声 $z$ 的输出，$p_z$​是噪声分布。
    - **判别器对真实样本的预测**：$\mathbb{E}_{x \sim p_{\text{data}}(x)} [\log D(x)]$
      - 对于每个真实样本，我们希望判别器输出 D(x)越接近 1 越好，因此需要最大化这部分的值。
    - **判别器对生成样本的预测**：$\mathbb{E}_{z \sim p_z(z)} [\log (1 - D(G(z)))]$
      - 这一项表示生成样本 G(z) 是由生成器 G 从噪声 z 中生成的。判别器 D 的输出 D(G(z)) 表示生成样本 G(z) 是真实的概率。对于每个生成样本，我们希望 D(G(z)) 越接近 0 越好，因此对 1−D(G(z)) 取对数。我们希望最大化这部分的值。
  - 对抗目标
    - 判别器 D：希望最大化 $V(D, G)$，即同时最大化真实数据的正确判别概率（使 $D(x)$ 尽可能接近 1）和最小化生成数据的错误判别概率（使 $D(G(z))$ 尽可能接近 0）。在训练判别器时，我们固定生成器 G，最大化以下目标函数：
      $\max_DV(D,G)=\mathbb{E}_{x\sim p_{\mathrm{data}}(x)}[\log D(x)]+\mathbb{E}_{z\sim p_z(z)}[\log(1-D(G(z)))]$
    - 生成器 G：希望最小化 $V(D,G)$，即欺骗判别器，让判别器认为生成数据是真实数据（使 $D(G(z))$ 尽可能接近 1）。在训练生成器时，我们固定判别器 D，最小化以下目标函数：
      $\min_GV(D,G)=\mathbb{E}_{z\sim p_z(z)}[\log(1-D(G(z)))]$
  - 训练过程
    ![image-20240724145611345](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240724145611345.png)
- GAN 和 DCGAN

  - GAN

    - 使用多层感知器（MLP）作为生成器和判别器。

    - 输入为一维向量（通常是从标准正态分布中采样的噪声）。

    - 输出也为一维向量（在处理图像时，需要将输出向量转换为图像格式）。
  - DCGAN
    - 使用卷积神经网络（CNN）作为生成器和判别器。
    - 输入为多维张量（通常为噪声向量的多维表示）。
    - 输出为多维张量（图像格式）。
    - ConvTranspose2d（反卷积或转置卷积）
      ![【深度学习反卷积】反卷积详解, 反卷积公式推导和在 Tensorflow 上的应用](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/v2-b41d087dccf8eefda38eadd4ec7bf0d2_1440w.image)
      - 是用于上采样的一种卷积运算，用于将低维度的潜在空间向高维度的图像空间进行映射。
      - 生成器网络使用了多个 `ConvTranspose2d` 层将一个潜在的噪声向量逐步上采样成一个指定大小的图像
      - `nn.ConvTranspose2d(padding=1, output_padding=1)`
        - padding：反卷积的填充与卷积相反，实际的 $p^′=k-1-p$
        - output_padding：在输出特征图的每一边的隐式填充

## 项目构建

### 数据集

- 数据集获取
  - [anime96 数据集](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.2/faces.zip) 5.1K+动漫头像 分辨率 96x96
  - 其他数据集 [GitHub - anime-face-dataset](https://github.com/learner-lu/anime-face-dataset)
    - 2.7K+  256x256
    - 14w+  512x512
  - 爬虫
    - 使用 [GAN 学习指南：从原理入门到制作生成 Demo](https://zhuanlan.zhihu.com/p/24767059) 中的爬虫代码爬取网站中的动漫图片
    - 使用 [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) 项目提取图片中的头像

- 数据集加载
  - 使用 ImageFolder 载入数据集

  - 对图片做标准化处理

  ```python
  class FaceDataset(Dataset):
      def __init__(self, data_path, image_size):
          transform = transforms.Compose([
              transforms.Resize(image_size),
              transforms.CenterCrop(image_size),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
          ])
          
          self.data = tv.datasets.ImageFolder(data_path, transform=transform)
  
      def __getitem__(self, index):
          return self.data.__getitem__(index)
      
      def __len__(self):
          return len(self.data)
  ```

### 网络定义

![image-20240724164804627](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240724164804627.png)

- 生成器定义

  ```python
  class NetG(nn.Module):
      def __init__(self, nz, ngf):
          super(NetG, self).__init__()
  
          self.main = nn.Sequential(
              # 输入是一个 nz 维度的噪声，我们可以认为它是一个 1 *1* nz 的 feature map
              nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
              nn.BatchNorm2d(ngf * 8),
              nn.ReLU(True),
              # 上一步的输出形状：(ngf*8) x 4 x 4
  
              nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ngf * 4),
              nn.ReLU(True),
              # 上一步的输出形状： (ngf*4) x 8 x 8
  
              nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ngf * 2),
              nn.ReLU(True),
              # 上一步的输出形状： (ngf*2) x 16 x 16
  
              nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ngf),
              nn.ReLU(True),
              # 上一步的输出形状：(ngf) x 32 x 32
  
              nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
              nn.Tanh()  # 输出范围 -1~1 故而采用 Tanh
              # 输出形状：3 x 96 x 96
          )
  
      def forward(self, input):
          return self.main(input)
  ```

- 判别器

  ```python
  class NetD(nn.Module):
      def __init__(self, ndf):
          super(NetD, self).__init__()
          
          self.main = nn.Sequential(
              # 输入 3 x 96 x 96
              nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
              nn.LeakyReLU(0.2, inplace=True),
              # 输出 (ndf) x 32 x 32
  
              nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ndf * 2),
              nn.LeakyReLU(0.2, inplace=True),
              # 输出 (ndf*2) x 16 x 16
  
              nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ndf * 4),
              nn.LeakyReLU(0.2, inplace=True),
              # 输出 (ndf*4) x 8 x 8
  
              nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
              nn.BatchNorm2d(ndf * 8),
              nn.LeakyReLU(0.2, inplace=True),
              # 输出 (ndf*8) x 4 x 4
  
              nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
              nn.Sigmoid()  # 输出一个数(概率)
          )
  
      def forward(self, input):
          return self.main(input).view(-1)
  ```

- 卷积层的偏置项和批量归一化的平移项都会对激活值进行平移，因此同时使用它们会造成参数的冗余。通过设置 `bias=False`，可以减少模型的参数数量，降低内存使用，并提高计算效率，同时不会影响模型的性能。

## 训练过程

```python

dataset = FaceDataset(data_path, image_size)
dataloader = t.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        drop_last=True
                                    )

# 网络
netg = NetG(nz, ngf)
netd = NetD(ndf)
netd.to(device)
netg.to(device)


# 定义优化器和损失
optimizer_g = t.optim.Adam(netg.parameters(), lr1, betas=betas)
optimizer_d = t.optim.Adam(netd.parameters(), lr2, betas=betas)
criterion = t.nn.BCELoss().to(device)

# 真图片 label 为 1，假图片 label 为 0
# noises 为生成网络的输入
true_labels = t.ones(batch_size).to(device)
fake_labels = t.zeros(batch_size).to(device)

fix_noises = t.randn(batch_size, nz, 1, 1).to(device)
noises = t.randn(batch_size, nz, 1, 1).to(device)

errord_meter = AverageValueMeter()
errorg_meter = AverageValueMeter()


epochs = range(epoch)
for epoch in iter(epochs):
    for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
        real_img = img.to(device)

        if ii % d_every == 0:
            # 训练判别器
            optimizer_d.zero_grad()
            ## 尽可能的把真图片判别为正确
            output = netd(real_img)
            error_d_real = criterion(output, true_labels)
            error_d_real.backward()

            # 尽可能把假图片判别为错误
            noises.data.copy_(t.randn(batch_size, nz, 1, 1))
            fake_img = netg(noises).detach()  # 根据噪声生成假图
            output = netd(fake_img)
            error_d_fake = criterion(output, fake_labels)
            error_d_fake.backward()
            optimizer_d.step()

            error_d = error_d_fake + error_d_real

            errord_meter.add(error_d.item())

        if ii % g_every == 0:
            # 训练生成器
            optimizer_g.zero_grad()
            noises.data.copy_(t.randn(batch_size, nz, 1, 1))
            fake_img = netg(noises)
            output = netd(fake_img)
            error_g = criterion(output, true_labels)
            error_g.backward()
            optimizer_g.step()
            errorg_meter.add(error_g.item())

    print(f'Epoch {epoch}: error_g={errorg_meter.value()[0]}, error_d={errord_meter.value()[0]}')
    errord_meter.reset()
    errorg_meter.reset()

    if (epoch+1) % save_every == 0:
        # 保存模型、图片
        fix_fake_imgs = netg(fix_noises)
        tv.utils.save_image(fix_fake_imgs.data[:64],
                            '%s/%s.png' % (save_path, epoch),
                            normalize=True)
        t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
        t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)

```

***

# Ch8 神经网络风格迁移

## 风格迁移概述

- 风格迁移又称风格转换，直观的类比就是给输入的图像加个滤镜，但是又不同于传统滤镜。风格迁移基于人工智能，每个风格都是由真正的艺术家作品训练、创作而成。只需要给定原始图片，并选择艺术家的风格图片，就能把原始图片转化成具有相应艺术家风格的图片。

- 风格迁移涉及三个主要图像

  - 内容图像（Content Image）：你希望保留其内容的图像。
  - 风格图像（Style Image）：你希望应用其风格的图像。
  - 生成图像（Generated Image）：通过优化生成的图像，它具有内容图像的内容和风格图像的风格。
  - 如下图，给定一张风格图片（左上角，手绘糖果图）和一张内容图片（右上角，斯坦福校园图），神经网络能够生成手绘风格的斯坦福校园图。
    ![image-20240724191059977](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240724191059977.png)

- 风格迁移通常使用预训练的卷积神经网络（如 VGG 网络）来提取图像的特征。这些网络在大型图像数据集上进行预训练，能够很好地捕捉图像的高层次和低层次特征。

- Neural Style 和 Fast Neural Style

    - Neural Style 使用了预训练的 VGG 网络来提取图像的内容特征和风格特征，从一个噪声开始计算内容损失和风格损失，通过梯度下降调整图片像素值。该方法生成图片效果好，但计算开销过大。
    - Fast Neural Style 专门设计了一个网络用来进行风格迁移，使生成过程大大加速，但需要为每种目标风格单独训练一个风格转换网络。
      

-  Fast Neural Style（快速神经风格迁移网络）
    ![image-20240729185844133](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240729185844133.png)

    - 网络构成：

      - 损失网络（Loss Network）：采用预训练的VGG网络提取内容图像、风格图像、生成图像的特征用于计算网络损失
        - VGG网络是一个经典的卷积神经网络，深度通常为16和19层（VGG-16、VGG-19），他的核心设计是选择使用3x3卷积核
          ![image-20240730131342708](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240730131342708.png)
          - 一个7\*7卷积核和堆叠3个3\*3卷积核可以得到相同的感受野。
          - 但是一个7\*7卷积核所需参数$=C \times 7 \times 7= 49C$，3个3\*3卷积核所需参数$=3 \times (C \times 3 \times 3)= 27C$
          - 同时，更多的卷积过程，特征提取更细致，非线性变换也会更多
      - 生成网络（$f_W$）：训练一个卷积神经网络来生成风格化的图像。这个网络接收内容图像作为输入，并输出风格化的图像。
    
    - 目标函数
    
      - 风格迁移的核心是定义一个目标函数，该函数衡量生成图像与内容图像和风格图像之间的差异。目标函数由两个部分组成：内容损失和风格损失。
    
        - **内容损失（Content Loss）**： 内容损失用于确保生成图像的内容与内容图像相似。通常选择损失网络中某一层的特征图来计算内容损失。
    
          $\mathcal{L}_{\text{content}} = \frac{1}{2} \sum_{i,j} (F_{ij}^{\text{generated}} - F_{ij}^{\text{content}})^2$

          - 其中，$F_{ij}$ 表示在预训练网络的某一层上，第 i 个特征图中的第 j 个像素。
    
        - **风格损失（Style Loss）**： 风格损失用于确保生成图像的风格与风格图像相似。通常选择预训练网络的多层特征图来计算风格损失，并使用格拉姆矩阵（Gram Matrix）来捕捉风格特征。
          $\mathcal{L}_{\text{style}} = \sum_l w_l \cdot \frac{1}{4N_l^2M_l^2} \sum_{i,j} (G_{ij}^l - A_{ij}^l)^2$
    
          - 其中，$G^l$ 和 $A^l$ 分别是生成图像和风格图像在第 l 层的格拉姆矩阵，$N_l$ 是第 l 层特征图的数量，$M_l$ 是第 l 层特征图的每个特征图的大小，$w_l$​ 是层的权重。
          - 格拉姆矩阵（Gram 矩阵）计算了卷积特征图之间的内积，反映了特征图之间的相对关系和空间分布，用于捕捉图像的风格特征。
    
            - 设输入特征图 y 的形状为 (B, C, H, W)，其中 B 是批次大小，C 是通道数，H 是高度，W 是宽度。
    
              - 将特征图 y 展平为一维张量 F，形状为 $(B, C, H\times W)$。$F_{i,j}$​ 表示第 i 个批次，第 j 个通道特征图的张量。
              - 计算 Gram 矩阵 G，形状为 $(B, C, C)$。
                $G_i = \frac{1}{C \times H \times W} F_i F_i^T$
              - $G_i$​ 表示第 i 个批次的 Gram 矩阵。
    
        - **总损失（Total Loss）**： 总损失是内容损失和风格损失的加权和。$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{content}} + \beta \cdot \mathcal{L}_{\text{style}}$
    
          - 其中，α 和 β 是超参数，控制内容损失和风格损失的相对重要性。
    
    - 训练过程：
      ![image-20240729185844133](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240729185844133.png)
    
      - 输人一张图片 x 到 $f_w$ 中得到结果 $\hat{y}$。
      - 将 $\hat{y}$ 和 $x$ 输入到 loss network（VGG-16）中，计算它在 relu3_3 的输出，并计算它们之间的均方误差作为 content loss。
      - 将 $\hat{y}$ 和 $y_s$ 输入到 loss network 中，计算它在 relu1_2、relu2_2、relu3_3 和 relu4_3 的输出，再计算它们的 GramMatrix 的均方误差作为style loss。
      - 两个损失相加，并反向传播，更新 $f_w$ 的参数，循环直到模型收敛

## 网络实现

- 数据预处理：

  - [COCO2014train](http://images.cocodataset.org/zips/train2014.zip) 数据集

    - 12.5G 8W+图片 多种分辨率
    - 对应提供了三种标注json文件
      - captions：图像描述的标注文件
      - instances：目标检测与实例分割的标注文件
      - person_keypoints：人体关键点检测的标注文件
    - COCO（Common Objects in Context）是由Microsoft发布并在计算机视觉领域被广泛应用和引用的数据集，旨在促进对象检测、分割和场景理解任务的研究。该数据集包含了丰富的图像和注释，涵盖了不同种类的对象和复杂的背景。

  - 加载并预处理内容图像和风格图像。

    ```python
    transfroms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 255)
    ])
    
    tv.datasets.ImageFolder(data_root, transfroms)
    ```

- 定义图像风格迁移网络：图像变换网络是一种卷积神经网络，它能够将内容图像转换为带有特定风格的图像。

  - 下采样层：通过三次卷积操作逐步减少图像的分辨率。

    - 反射填充`nn.ReflectionPad2d(reflection_padding=3)`：

      - 左右填充：
        <img src="https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240730160845233.png" alt="image-20240730160845233" style="zoom:80%;" />
      - 上下填充：
        ![image-20240730160905452](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240730160905452.png)

    - 使用反射填充的原因：

      - 在标准卷积核中使用的零填充，这可能引入不自然的边缘伪影，影响图像的自然性。

      - 反射填充通过在边界处反射输入的边缘像素来进行填充。这种填充方式比零填充更自然，因为它保持了边缘像素的连续性，避免了边缘伪影。

  - 残差层：通过五个残差块提高模型的表达能力。
  - 上采样层：通过上采样操作将图像恢复到原始分辨率。

    - `torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")`
      - 假设 x 的形状为 `(B, C, H, W)`
      - 函数会将特征图的大小扩大至`scale_factor`倍
      - 然后根据`mode="nearest"`将最近的特征值复制的新位置


  ```python
  class TransformerNet(nn.Module):
      def __init__(self):
          super(TransformerNet, self).__init__()
  
          # Down sample layers
          self.initial_layers = nn.Sequential(
              ConvLayer(3, 32, kernel_size=9, stride=1),
              nn.InstanceNorm2d(32, affine=True),
              nn.ReLU(True),
              ConvLayer(32, 64, kernel_size=3, stride=2),
              nn.InstanceNorm2d(64, affine=True),
              nn.ReLU(True),
              ConvLayer(64, 128, kernel_size=3, stride=2),
              nn.InstanceNorm2d(128, affine=True),
              nn.ReLU(True),
          )
  
          # Residual layers
          self.res_layers = nn.Sequential(
              ResidualBlock(128),
              ResidualBlock(128),
              ResidualBlock(128),
              ResidualBlock(128),
              ResidualBlock(128)
          )
  
          # Upsampling Layers
          self.upsample_layers = nn.Sequential(
              UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
              nn.InstanceNorm2d(64, affine=True),
              nn.ReLU(True),
              UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
              nn.InstanceNorm2d(32, affine=True),
              nn.ReLU(True),
              ConvLayer(32, 3, kernel_size=9, stride=1)
          )
  
      def forward(self, x):
          x = self.initial_layers(x)
          x = self.res_layers(x)
          x = self.upsample_layers(x)
          return x
  
  class ConvLayer(nn.Module):
      def __init__(self, in_channels, out_channels, kernel_size, stride):
          super(ConvLayer, self).__init__()
          reflection_padding = int(np.floor(kernel_size / 2))
          self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
          self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
  
      def forward(self, x):
          out = self.reflection_pad(x)
          out = self.conv2d(out)
          return out
  
  class UpsampleConvLayer(nn.Module):
      def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
          super(UpsampleConvLayer, self).__init__()
          self.upsample = upsample
          reflection_padding = int(np.floor(kernel_size / 2))
          self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
          self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
  
      def forward(self, x):
          x_in = x
          if self.upsample:
              x_in = t.nn.functional.interpolate(x_in, scale_factor=self.upsample)
          out = self.reflection_pad(x_in)
          out = self.conv2d(out)
          return out
  
  
  class ResidualBlock(nn.Module):
      def __init__(self, channels):
          super(ResidualBlock, self).__init__()
          self.main = nn.Sequential(
                  ConvLayer(channels, channels, kernel_size=3, stride=1)
                  nn.InstanceNorm2d(channels, affine=True)
                  nn.ReLU()
                  ConvLayer(channels, channels, kernel_size=3, stride=1)
                  nn.InstanceNorm2d(channels, affine=True)
          )
  
      def forward(self, x):
          out = self.main(x) + x
          return out
  ```

- 定义损失网络：损失网络通常使用预训练的 VGG 网络，用于提取内容特征和风格特征。
  ![image-20240725113722082](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240725113722082-1722251805645-5.png)
  
  ```python
  class Vgg16(torch.nn.Module):
      def __init__(self):
          super(Vgg16, self).__init__()
          features = list(vgg16(pretrained=True).features)[:23]
          # the 3rd, 8th, 15th and 22nd layer of \ 
          # self.features are: relu1_2, relu2_2, relu3_3, relu4_3
          self.features = nn.ModuleList(features).eval()
  
      def forward(self, x):
          results = []
          for ii, model in enumerate(self.features):
              x = model(x)
              if ii in {3, 8, 15, 22}:
                  results.append(x)
  
          vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
          return vgg_outputs(*results)
  ```
  
  - 初始化
  
    - 从预训练的 VGG16 模型中提取前 23 层，并存储在 `self.features` 中。前 23 层包含了从输入图像到 `relu4_3` 层的所有卷积层和 ReLU 层。
    - 将这些层存储在一个 `list` 中，并设置为评估模式 。
  
  - 前向传播方法：
  
    - 输入 `x` 经过特征提取层，依次通过 `self.features` 中的每一层。
  
    - 在第 3 层、第 8 层、第 15 层和第 22 层（即 `relu1_2`，`relu2_2`，`relu3_3` 和 `relu4_3`）处，提取特征并存储在 `results` 列表中。
  
    - 使用 `namedtuple` 创建一个包含这些特征的元组，并返回该元组。
  
- 定义损失函数：

  - 内容损失：确保生成图像的内容与原内容图像相似。
    $\mathcal{L}_{\text{content}} = \frac{1}{2} \sum_{i,j} (F_{ij}^{\text{generated}} - F_{ij}^{\text{content}})^2$

    ```python
    features_y = vgg(transformer(x))
    features_x = vgg(x)
    content_loss = F.mse_loss(features_y.relu2_2, features_x.relu2_2)
    ```

    - 为什么是 relu2_2
      - relu2_2 层的感受野相比于更浅层的 relu1_2 层更大，能够捕捉更大范围的图像特征，但相比于更深层的 relu3_3 和 relu4_3 层，感受野又较小，能够保留更多的细节。在细节和抽象之间达到了一个较好的平衡，可以在保留足够多细节信息的同时，捕捉到一定程度的抽象和全局特征。

  - 风格损失：确保生成图像的风格与风格图像相似。
    $\mathcal{L}_{\text{style}} = \sum_l w_l \cdot \frac{1}{4N_l^2M_l^2} \sum_{i,j} (G_{ij}^l - A_{ij}^l)^2$​

    ```python
    with t.no_grad():
        features_style = vgg(style)
        gram_style = [utils.gram_matrix(y) for y in features_style]
    
    features_y = vgg(transformer(x))
    features_x = vgg(x)
    for ft_y, gm_s in zip(features_y, gram_style):
    	gram_y = utils.gram_matrix(ft_y)
        # y 是一个 batch 的生成图片，style 是一个风格图片，所以需要 expand_as
    	style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
    ```

  - 总变分损失：使用总损失更新参数

    ```python
    total_loss = content_weight * content_loss + style_weight * style_loss
    total_loss.backward()
    optimizer.step()
    ```

- 训练网络：

  - 使用内容图像和风格图像训练图像变换网络，通过优化总损失函数更新风格迁移网络的参数。
  - 训练完成后，使用 `t.save(transformer.state_dict(), )` 保存模型

- 生成图像：训练完成后，使用图像变换网络快速生成带有特定风格的图像

  ```python
  content_image = tv.datasets.folder.default_loader(content_path)
  content_transform = tv.transforms.Compose([
      tv.transforms.ToTensor(),
      tv.transforms.Lambda(lambda x: x.mul(255))
  ])
  content_image = content_transform(content_image)
  content_image = content_image.unsqueeze(0).to(device).detach()
  
  style_model = TransformerNet().eval()
  style_model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
  style_model.to(device)
  
  output = style_model(content_image)
  output_data = output.cpu().data[0]
  tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)
  ```

***

# Ch10 图像描述Image Caption

## ImageCaption

- ImageCaption，通常被翻译为图像描述，也有人称之为图像标注，直观地解释就是从给定的图像生成一段描述文字。
- 模型构建
  ![image-20240805144731007](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240805144731007.png)
  - 训练过程：
    - 图片经过神经网络提取到图片高层次的语义信息$f$
    - 将$f$输人到LSTM中，并希望LSTM的输出是$S_0$
    - 将$S_0$输入到LSTM中，并希望LSTM的输出是$S_1$
    - ……
    - 将$S_{N-1}$输入到LSTM中，并希望LSTM的输出是$S_N$
      - $S_0$, $S_1$, ... ,$S_N$ 是对图片进行描述的语句

## 数据处理

- 数据集

  - [AI Challenger图像中文描述数据集](https://tianchi.aliyun.com/dataset/145781/)

  - 数据来自2017 AI Challenger，数据集对给定的每一张图片有五句话的中文描述。数据集包含30万张图片，150万句中文描述。训练集：210,000 张，验证集：30,000 张，测试集 A：30,000 张，测试集 B：30,000 张。

  - 数据集中包含图片文件与图像描述文件`annotations.json`
    ```json
    {
        "caption": [
            "两个衣着休闲的人在平整的道路上交谈",
            "一个穿着红色上衣的男人和一个穿着灰色裤子的男人站在室外的道路上交谈",
            "室外的公园里有两个穿着长裤的男人在交流",
            "街道上有一个穿着深色外套的男人和一个穿着红色外套的男人在交谈",
            "道路上有一个身穿红色上衣的男人在和一个抬着左手的人讲话"
        ],
        "image_id": "8f00f3d0f1008e085ab660e70dffced16a8259f6.jpg",
        "url": "http://m4.biz.itc.cn/pic/new/n/71/65/Img8296571_n.jpg"
    },
    ```

    - caption：图片对应的五句中文描述
    - image_id：图片的文件名
    - url：图片的下载地址

- 数据预处理：

  - 描述预处理

    - 数据读入，使用json库读入描述文件，并取出需要的部分

      ```python
      with open('annotations.json') as file:
          data = json.load(file)
          
      id2ix = {item['image_id']: ix for ix, item in enumerate(data)}
      ix2id = {ix: id for id, ix in (id2ix.items())}
      
      captions = [item['caption'] for item in data]
      ```

    - 中文分词

      ```Python
      # 分词结果
      cut_captions = [[list(jieba.cut(sentence)) for sentence in sentences] for sentences in caption)]
      
      # 遍历 cut_captions，更新每个单词的出现次数
      word_nums = {}  # 用于存储每个单词及其出现次数
      for sentences in cut_captions:
      	for sentence in sentences:
              for word in sentence:
                  word_nums[word] = word_nums.get(word, 0) + 1
      ```

    - 将词用序号表示（word2ix），并过滤低频词

      ```Python
      word_nums_list = sorted([(num, word) for word, num in word_nums.items()], reverse=True)
      
      words = [word[1] for word in word_nums_list[:opt.max_words] if word[0] >= opt.min_appear]
      
      unknown = '</UNKNOWN>' # 表示在词汇表中未找到的单词。
      end = '</EOS>' # 表示句子的结束
      padding = '</PAD>' # 用于对序列进行填充
      words = [unknown, padding, end] + words
      
      word2ix = {word: ix for ix, word in enumerate(words)}
      ix2word = {ix: word for word, ix in word2ix.items()}
      
      ix_captions = [[[word2ix.get(word, word2ix.get(opt.unknown)) for word in sentence]
                      for sentence in item]
                     for item in cut_captions]
      ```

    - 保存处理结果

      ```Python
      results = {
          'caption': ix_captions,
          'word2ix': word2ix,
          'ix2word': ix2word,
          'ix2id': ix2id,
          'id2ix': id2ix,
          'padding': '</PAD>',
          'end': '</EOS>',
          'readme': readme
      }
      
      t.save(results, "caption.pth")
      ```

  - 图片预处理

    - 数据载入

      ```Python
      class CaptionDataset(data.Dataset):
          def __init__(self, caption_data_path):
              self.transforms = tv.transforms.Compose([
                  tv.transforms.Resize(256),
                  tv.transforms.CenterCrop(256),
                  tv.transforms.ToTensor(),
                  tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
              ])
      
              data = t.load(caption_data_path)
              self.ix2id = data['ix2id']
              # 所有图片的路径
              self.imgs = [os.path.join("data/imgs/", self.ix2id[_]) for _ in range(len(self.ix2id))]
      
          def __getitem__(self, index):
              img = Image.open(self.imgs[index]).convert('RGB')
              img = self.transforms(img)
              return img, index
      
          def __len__(self):
              return len(self.imgs)
      ```

    - 特征提取

      ```Python
      # 200000*2048 20万张图片，每张图片2048维的feature
      results = t.Tensor(len(dataloader.dataset), 2048).fill_(0)
      
      resnet50 = tv.models.resnet50(pretrained=True)
      resnet50.fc = lambda x: x
      
      for ii, (imgs, indexs) in enumerate(dataloader):
          features = resnet50(imgs)
          results[ii * batch_size:(ii + 1) * batch_size] = features.data.cpu()
      
      
      t.save(results, 'results.pth')
      ```
    
  - 数据集载入
  
    - 数据集定义
  
        ```Python
        class CaptionDataset(data.Dataset):
            def __init__(self, caption_data_path, img_feature_path):
                self.opt = opt
                data = t.load(caption_data_path)
                word2ix = data['word2ix']
                self.captions = data['caption']
                self.padding = word2ix.get(data.get('padding'))
                self.end = word2ix.get(data.get('end'))
                self._data = data
                self.ix2id = data['ix2id']
                self.all_imgs = t.load(img_feature_path)
        
            def __getitem__(self, index):
                img = self.all_imgs[index]
        
                caption = self.captions[index]
                # 5句描述随机选一句
                rdn_index = np.random.choice(len(caption), 1)[0]
                caption = t.LongTensor(caption[rdn_index])
                
                return img, caption, index
        
            def __len__(self):
                return len(self.ix2id)
        ```
  
    - 将所有描述补齐到等长（在`dataloader`中自定义`collate_fn`）
  
        - 输入：一个batch_size的数据，形式为`[(img1, cap1, index1), (img2, cap2, index2) ....]`
        - 拼接策略如下：
            - 选取长度最长的句子（长度=句子长度+结束标记）作为该批次的最大长度（batch_length）
            - 创建大小为$\text{batch\_length} \times \text{batch\_size}$使用`pad`标记填充的张量
            - 将每个描述复制到 `cap_tensor` 中，在结尾添加 `eos` 标记。
        - 输出：`(imgs, (cap_tensor, lengths), indexs)`
            - `imgs(Tensor)`: batch_sie * 2048
            - `cap_tensor(Tensor)`: max_length * batch_size
            - `lengths(list of int)`: 长度为batch_size
            - `index(list of int)`: 长度为batch_size
        
        ```python
        dataloader = data.DataLoader(dataset, batch_size=, shuffle=True, num_workers=4, collate_fn=create_collate_fn(dataset.padding, dataset.end))
        
        def create_collate_fn(padding, eos, max_length=50):
            def collate_fn(img_cap):
                img_cap.sort(key=lambda p: len(p[1]), reverse=True)
                imgs, caps, indexs = zip(*img_cap)
                imgs = t.cat([img.unsqueeze(0) for img in imgs], 0)
                lengths = [min(len(c) + 1, max_length) for c in caps]
                batch_length = max(lengths)
                cap_tensor = t.LongTensor(batch_length, len(caps)).fill_(padding)
                for i, c in enumerate(caps):
                    end_cap = lengths[i] - 1
                    cap_tensor[end_cap, i] = eos
                    cap_tensor[:end_cap, i].copy_(c[:end_cap])
                return (imgs, (cap_tensor, lengths), indexs)
        
            return collate_fn
        
        ```
  
- 模型训练

  - 模型定义

    ```Python
    class CaptionModel(nn.Module):
        def __init__(self, embedding_dim, lstm_hidden, num_layers, word2ix, ix2word):
            super(CaptionModel, self).__init__()
            self.ix2word = ix2word
            self.word2ix = word2ix
            
            self.fc = nn.Linear(2048, opt.lstm_hidden)
            self.embedding = nn.Embedding(len(word2ix), opt.embedding_dim)
    
            self.lstm = nn.LSTM(embedding_dim, lstm_hidden, num_layers=num_layers)
            self.classifier = nn.Linear(lstm_hidden, len(word2ix))
    
        def forward(self, img_feats, captions, lengths):
            embeddings = self.embedding(captions)
            
            # img_feats是2048维的向量,通过全连接层转为256维的向量,和词向量一样
            img_feats = self.fc(img_feats).unsqueeze(0)
            
            # 将img_feats看成第一个词的词向量 
            embeddings = t.cat([img_feats, embeddings], 0)
            
            # PackedSequence
            packed_embeddings = pack_padded_sequence(embeddings, lengths)
            outputs, state = self.lstm(packed_embeddings)
            # lstm的输出作为特征用来分类预测下一个词的序号
            # 因为输入是PackedSequence,所以输出的output也是PackedSequence
            
            pred = self.classifier(outputs[0])
            return pred, state
        
        def generate(self, img, eos_token='</EOS>',
                     beam_size=3,
                     max_caption_length=30,
                     length_normalization_factor=0.0):
            
            cap_gen = CaptionGenerator(embedder=self.embedding,
                                       rnn=self.lstm,
                                       classifier=self.classifier,
                                       eos_id=self.word2ix[eos_token],
                                       beam_size=beam_size,
                                       max_caption_length=max_caption_length,
                                       length_normalization_factor=length_normalization_factor)
            
            img =img.unsqueeze(0)
            img = self.fc(img).unsqueeze(0)
            sentences, score = cap_gen.beam_search(img)
            sentences = [' '.join([self.ix2word[idx] for idx in sent])
                         for sent in sentences]
            return sentences
    
    ```
    
    - beam_search

      - Beam Search 是一种启发式图搜索算法，Beam Search 的核心思想是维护一个固定大小的候选列表（称为 beam），在每一步中，算法只保留最有可能的几个候选节点，而不是考虑所有可能的节点。这个“最有可能”的判断通常基于节点的累积得分，该得分是节点从起始点到当前节点路径的得分之和。
      - Beam Search 类似于 BFS（广度优先搜索），但它有以下关键区别和限制：
        - 限制宽度（束宽度，beam width）：
           - 在广度优先搜索中，所有可能的路径都被探索，而 Beam Search 仅保留具有最高得分的 `beam_size` 条路径。这意味着在每个时间步中，只会追踪 `beam_size` 个得分最高的部分描述（partial captions）。
        - 描述得分的计算：
           - 每个部分描述的得分通是各个词的对数概率的加权和。在许多实现中，这些对数概率的和可以直接作为得分，但有时为了平衡描述的长度和得分，会加入长度归一化因子。
             $score = \frac{1}{\text{len(sentence)}^\alpha} \sum_{i=1}^{\text{len(sentence)}} \text{softmax}(P(w_i))$
             - 其中，`len(sentence)` 是描述的长度，$\alpha$ 是长度归一化因子。如果 $\alpha > 0$，则长句子会得到更低的惩罚，防止过度偏好短句子。$\log P(w_i)$ 是每个词的对数概率。
        - 替换较低得分的描述：
           - 在生成新词并更新部分描述后，Beam Search 只保留最高得分的 `beam_size` 个部分描述，其余的则会被丢弃。这类似于一个优先队列，其中总是保留得分最高的元素。
      - 基本步骤：
        ![3275fb59ace74e129ab01c94372d9fd3](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/3275fb59ace74e129ab01c94372d9fd3.png)
        - 将图片特征作为LSTM的初始输入得到k个最有可能的下一位置word，并加入到候选列表中；
        - 对于候选列表中的每个序列，生成k个最有可能的后继节点，与前缀节点拼接后加入候选序列；
        - 在加入新序列的过程中，计算每个描述的得分并只保留得分最高的k个描述；
        - 当生成的后继节点为终止标记或时，将序列加入完整描述序列；
        - 重复以上步骤，直到达到序列的最大长度，或者候选列表中没有新的节点生成。
        - 选择最终结果：从完整描述序列中选择得分最高的节点作为搜索结果。
      
    
  - 模型训练
  
    ```Python
    def train():
        caption_data_path = 'caption.pth'
        img_feature_path = 'results.pth' 
    
        dataset = CaptionDataset(caption_data_path, img_feature_path)
        dataloader = data.DataLoader(dataset,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=4,
                                     collate_fn=create_collate_fn(dataset.padding,dataset.end))
        word2ix = dataloader.dataset._data['word2ix']
        ix2word = dataloader.dataset._data['ix2word']
    
        model = CaptionModel(opt, word2ix, ix2word)
        optimizer = model.get_optimizer(opt.lr)
        criterion = t.nn.CrossEntropyLoss()
    
        loss_meter = meter.AverageValueMeter()
    
        for epoch in range(opt.epoch):
            loss_meter.reset()
            for ii, (imgs, (captions, lengths), indexes) in tqdm.tqdm(enumerate(dataloader)):
                optimizer.zero_grad()
                captions = captions.to(device)
                input_captions = captions[:-1]
                target_captions = pack_padded_sequence(captions, lengths)[0]
                score, _ = model(imgs, input_captions, lengths)
                loss = criterion(score, target_captions)
                loss.backward()
                optimizer.step()
                loss_meter.add(loss.item())
            print(f"Epoch: {epoch}, Loss: {loss_meter.value()}")
    ```
  
- 描述生成

  ```Python
  def generate():
      # 数据预处理
      data = t.load(opt.caption_data_path, map_location=lambda s, l: s)
      word2ix, ix2word = data['word2ix'], data['ix2word']
  
      transforms = tv.transforms.Compose([
          tv.transforms.Resize(opt.scale_size),
          tv.transforms.CenterCrop(opt.img_size),
          tv.transforms.ToTensor(),
          tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
      img = Image.open(opt.test_img)
      img = transforms(img).unsqueeze(0)
  
      # 用resnet50来提取图片特征
      resnet50 = tv.models.resnet50(True).eval()
      resnet50.fc = lambda x: x
      img_feats = resnet50(img).detach()
  
      # 使用Caption模型的generate方法生成描述
      model = CaptionModel(opt, word2ix, ix2word)
      model = model.load(opt.model_ckpt).eval()
      model.to(device)
  
      results = model.generate(img_feats.data[0])
      print('\r\n'.join(results))
  ```
  
  - 生成结果
    ![caption-results](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/caption-results.png)
  

***

# DeepLab V3+

- 模型框架
  <img src="https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240731092320864.png"  />

  - 主干网络DCNN：

    - MobileNetV2

      - MobileNet模型是Google针对手机等嵌入式设备提出的一种轻量级的深层神经网络，MobileNetV2是MobileNet的升级版，它具有一个非常重要的特点就是使用了Inverted resblock（逆残差），整个mobilenetv2主题结构都由Inverted resblock组成。
      - 残差与逆残差
        ![c915a2464ef425f96d4c0926737549a8](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/c915a2464ef425f96d4c0926737549a8.png)
        - 普通残差是先使用`1*1`卷积降维，再进行`3*3`卷积，最后使用`1*1`卷积升维(两边厚，中间薄)
        - 逆残差块是先使用`1*1`卷积进行升维，然后使用`3*3`深度可分离卷积，最后使用`1*1`卷积进行降维(两边薄，中间厚)
          - 逆残差输入和输出是低维，中间处理是高维，大部分重要计算在扩展的高维空间进行，但输入输出是低维，因此更节省计算资源适合移动和嵌入式设备
      - Inverted resblock
        ![3d92dd155c06bd46fe9525f1d212453a](https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/3d92dd155c06bd46fe9525f1d212453a.png)
        - Expand阶段：如果 `expand_ratio > 1`，会先通过 1x1 卷积将输入通道数扩展为 `input_channel * expand_ratio`。
          - ReLU6是ReLU激活函数的一个变体，在低精度浮点数下有比较好的表示性能，适用于移动设备$f(x) = min(max(0, x), 6)$

        - Depthwise卷积阶段：进行 3x3 深度卷积
          - `nn.Conv2d(groups=)`组卷积，`groups` 参数决定了输入和输出通道如何被分组来进行卷积运算。
            - 如果 `groups=1`，则所有的输入和输出通道都会参与卷积运算（标准的卷积）。
            - 如果 `groups=C_in`，则每一组包含一个输入通道和相应的输出通道，这相当于深度可分离卷积中的深度卷积（depthwise convolution）部分。
            - 如果 `groups > 1` 但不是 `C_in`，则输入和输出通道会被分成多个组，每组内部进行独立的卷积运算，不同组之间没有权重共享。

        - Projection阶段：最后通过 1x1 卷积将通道数降回输出通道数 `output_channel`。
        - 残差连接：如果输入和输出的形状相同且步幅为 1，则进行残差连接，即跳跃连接输入和输出。

      - MobileNetv2
        - `width_mult`：用于调整通道数的缩放系数。默认值为 1.0。
        - 初始卷积层`conv_bn`
        - 多个 `InvertedResidual` 模块对特征图进行下采样特征提取
        - 分类器部分 `classifier`（后续不会使用）
          - 全局平均池化：对特征图的空间维度（宽度和高度）进行平均池化，将特征图转换为一个单一的特征向量。
          - Dropout：在全连接层之前应用 Dropout (p=0.2)，以防止过拟合。
          - 全连接层：将特征向量输入到一个全连接层中，输出 1000 维的向量，表示对 1000 个类别的预测。
        - 模型初始化 (`_initialize_weights`)
          - 使用正态分布初始化卷积层和全连接层的权重。
          - Batch Normalization 层的权重初始化为 1，偏置初始化为 0。
        - deeplabv3中对mobilenetv2的使用
          - 特征提取 (`self.features`)：使用MobileNetV2的特征提取部分（去掉最后一个1x1卷积层），用于提取输入图像的中间特征。
          - 降采样因子 (`downsample_factor`)：决定特征图的下采样倍数。
            - `self.down_idx = [2, 4, 7, 14]`定义了一组用于下采样的层索引，即在 MobileNetV2 中，哪些层会降低特征图的空间分辨率。
            - 操作：根据 downsample_factor 的值，调整特定层的膨胀率（dilation rate）和步幅（stride）。
              - 当 downsample_factor=8 时：输出特征图的大小是输入图像的 1/8。
                - 第 `self.down_idx[-2]` 到 `self.down_idx[-1]`之间的层：将步幅设置为1，并设置膨胀率为2。
                - 第 `self.down_idx[-1]` 到模型结束的层：将步幅设置为1，并设置膨胀率为4。
              - 当 downsample_factor=16 时：
                - 第 `self.down_idx[-1]` 到模型结束的层：将步幅设置为1，并设置膨胀率为2。
            - 操作函数 (`_nostride_dilate`)：检查每一个卷积层（`Conv`层），如果步幅为2且核大小为3x3，则将步幅修改为1，并根据给定的 `dilate` 参数调整膨胀率和填充大小。
        - 前向传播调整：
          - 前4个特征层提取低层特征 (`low_level_features`)，而剩下的特征层提取更高层的特征。
          - `return low_level_features, x `

    - Xception

      - SeparableConv2d 模块

        - 深度可分离卷积的实现。
          - 深度卷积：
          - 逐点卷积：
        - activate_first
          - 在卷积前应用激活限制输入的范围，有助于增加稀疏性。
          - 在卷积后应用激活是更传统的方法，允许网络学习更广泛的特征。

      - Block 模块

        - 膨胀卷积：通过 `atrous` 参数指定的膨胀率应用于每个可分离卷积层。
        - 跳跃连接：
          - 如果输入通道数 `in_filters` 与输出通道数 `out_filters` 不同，或卷积步幅 `strides` 不等于 1，使用 1x1 卷积调整输入特征图的尺寸。（相同则不做处理）
          - `self.skip` 和 `self.skipbn` 定义了这个跳跃连接的卷积层和批归一化层。
          - `self.head_relu = True`？
            - 控制SeparableConv2d 模块中第一个relu的inplace参数值
            - 如果输入输出通道数相同，那么`skip=inp`，如果第一个relu如果是`inplace=false`则会改变skip的值
            - 但是，代码好像写反了，或者根本不是这个原因
        - 可分离卷积（Separable Convolution）：
          - 根据 `grow_first` 参数决定通道数是在第一个可分离卷积提升还是在第二个
            - 在更早的阶段增加通道数有助于提取更加复杂和多样的特征，但也会导致后续卷积操作的计算量增加
          - `self.sepconv3`：第三个可分离卷积根据 `strides` 改变特征图的空间维度做下采样。

      - Xception

        - 初始卷积层

          - 输入的图像首先通过一个 3x3 的卷积和 1x1 的卷积提取初步的低层次特征。

        - 主干特征提取模块

          - 包含多个 `Block` 模块。根据 `downsample_factor` 的不同，决定了下采样的步幅（strides）。

          ```python
          if downsample_factor == 8:
              stride_list = [2,1,1]
          elif downsample_factor == 16:
              stride_list = [2,2,1]
          
          self.block1 = Block(64, 128, 2)
          self.block2 = Block(128, 256, stride_list[0], inplace=False)
          self.block3 = Block(256, 728, stride_list[1])
          # ... 多个 block
          self.block20 = Block(728, 1024, stride_list[2], atrous=rate, grow_first=False)
          ```

        - 中间膨胀卷积模块

          - 中间部分（block4 - block19）使用了膨胀卷积来增加感受野，而不增加计算量。膨胀卷积的膨胀率由 `rate = 16 // downsample_factor` 决定。

          ```python
          rate = 16 // downsample_factor
          self.block4 = Block(728, 728, 1, atrous=rate)
          # ... 多个 block
          self.block20 = Block(728, 1024, stride_list[2], atrous=rate, grow_first=False)
          ```

        - 最终卷积层

          - 在最后的卷积层中，模型进一步提取高级特征。这些卷积层继续使用深度可分离卷积，并保持激活顺序。

          ```python
          self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)
          self.conv4 = SeparableConv2d(1536, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)
          self.conv5 = SeparableConv2d(1536, 2048, 3, 1, 1 * rate, dilation=rate, activate_first=False)
          ```

        - block2 的中间输出作为浅层特征 `low_feature_layer`，最终的深层特征输出为 `x`。

  - ASPP

    - 分支结构：
      - Branch 1: 1x1卷积，标准卷积，用于捕捉局部细节。
      - Branch 2-4: 3x3卷积，使用不同的膨胀率（6、12、18）来提取多尺度特征。
      - Branch 5: 全局平均池化，随后通过1x1卷积，生成全局上下文特征。
    - 特征融合 (`conv_cat`)：将所有分支的输出特征图拼接在一起，然后通过1x1卷积整合为一个特征图。

  - deeplabv3+
    <img src="https://raw.githubusercontent.com/Duangboss/Typora-Image/main/img/image-20240731092320864.png"  />



