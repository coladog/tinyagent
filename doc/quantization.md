# Integer-based computing in TinyAgent

## Quantization

[Quantization](https://en.wikipedia.org/wiki/Quantization) techniques are widely used in large neural network to reduce memory consumption and inference overhead.
Its idea is to use low-precision data types to express the model (for example, from 32-bits floating point to 8-bit integer), thus trading computational precision for resource overhead.

In TinyAgent, the goal is to integrate the computation so that it can run in environments that do not support floating point computation.
We use a relatively simple and basic quantization strategy, and more advanced techniques can be used to further reduce overhead and tune the computing accuracy.


The basic equation is:
$$r = S(q − Z)$$
where $r$ is the real number, $q$ is the quantized number, $S$ is the scaling factor and $Z$ is the zero-point factor.
$S$ and $Z$ are two main constants and the main quantization parameters.
$S$ is a floating-point number that expresses the interval of floating-point numbers that a level can represent.
$Z$ is the quantized value of zero.

For example, let $r \in [-1, 1]$ and $q$ is represented in 8-bit  unsigned integer.
We can let $Z$ be 127, let [0, 127] represent [-1, 0] and let [127, 254] represent [0, 1].
In this way $S = \frac{1}{127} \approx 0.0079$.
When $r = 0.23$, the corresponding $q = \lfloor \frac{0.23}{0.0079} \rfloor + 127 = 156$.

TinyAgent implements the basic operations that support matrix multiplication: addition and multiplication.
Also, it supports integrated implementations of three activation functions: ReLU, Tanh, and Sigmoid.
We next describe the mathematical principles and implementations we use separately.

### Multiplication

Let $r_1, r_2$ and $r_3$ denote three real numbers, $r_4 = r_1 + r_2$.
Let $q_1, q_2$ and $q_3$ denote the quantized values of $r_1, r_2$ and $r_4$ using parameters $S$ and $Z$.
We have:

$$q_3 = \frac{r_3}{S} + Z = \frac{r_1r_2}{S} + Z = \frac{S(q_1-Z)S(q_2-Z)}{S} + Z = S(q_1q_2-Zq_2-Zq_1+Z^2) + Z$$


### Addition

Let $r_1, r_2$ and $r_4$ denote three real numbers, $r_3 = r_1 + r_2$.
Let $q_1, q_2$ and $q_4$ denote the quantized values of $r_1, r_2$ and $r_3$ using parameters $S$ and $Z$.
We have:

$$q_4 = \frac{r_4}{S} + Z = \frac{r_1+r_2}{S} + Z = \frac{S(q_1-Z) + S(q_2-Z)}{S} + Z = q_1 + q_2 - Z$$

### ReLU

The function of ReLU can be viewed as:
$$f(x) = max(x, 0)$$
In our implementation, the quantized computation can still be expressed as:
$$f(x_{quantized}) = max(x_{quantized}, 0_{quantized})$$

### [Tanh](https://paperswithcode.com/method/tanh-activation#:~:text=Tanh%20Activation%20is%20an%20activation,for%20multi%2Dlayer%20neural%20networks.) and [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)

These two activation functions involve complex calculations such as $e^x$.
In computing environments with floating-point support, such calculations are often performed by [series expansions](https://en.wikipedia.org/wiki/Taylor_series) to obtain high-precision approximate solutions.
We pre-store the inputs and outputs with certain precision and use a binary search on the table to find the outputs at query time.

### Implementation

In TinyAgent, we set:

* **The real value $r \in [-10, 10].$** This is a bold assumption: all values in the inference process will not exceed this range. In practice, the parameters of a trained neural network tend to be centrally distributed between [-1, 1], provided that the inputs and outputs are properly regularized, this setting was good enough for our tests.
* $Z = 0, S = 2^{-30}.$ In this case, the quantized data can be stored in 32-bit signed integers.

In this way, for multiplication, the formular can be simplified into:
$$q_3 = Sq_1q_2$$
The actual implementation in C can be written as:
$$q_3 = (int32) ((int64) q_1 * (int64) q_2) >> 30) $$
For addition, the formular can be simplified into:
$$q_4 = q_1 + q_2$$

