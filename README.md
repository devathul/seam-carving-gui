## Seam Carving (接缝裁接) 算法实现

本项目实现了Seam Carving算法，并提供了GUI界面。

**参考论文**：***Seam Carving for Content-Aware Image Resizing*** - *Shai Avidan, Ariel Shamir*

### 算法思想

在图像的能量谱(*energy map*)上，找一条能量和最低的接缝(*seam*)，接缝上所有的相邻像素点都互在对方的8联通域内。删除或增加这条接缝。反复执行这个过程，直到图像达到所需尺寸。

寻找接缝使用动态规划思想。以垂直方向接缝为例，转移方程：

$\textbf M(i,j)=e(i,j)+min\{\textbf M(i-1,j-1),\textbf M(i-1,j),\textbf M(i-1,j+1)\}$

### 能量谱的计算

本项目采用一对Sobel算子进行图像能量谱计算（梯度场计算）：
$$
\textbf u=
\left[
\begin{matrix}
+1 & +2 & +3 \\
0 & 0 & 0 \\
-1 & -2 & -1
\end{matrix}
\right]
$$
$$
\textbf v=
\left[
\begin{matrix}
-1 & 0 & +1 \\
-2 & 0 & +2 \\
-1 & 0 & +1
\end{matrix}
\right]
$$

将$\textbf u$与$\textbf v$分别与图像作卷积，两个卷积结果相加，即为图像的能量谱。

Scharr等算子可以替代Sobel算子来进行能量谱计算。在论文的"**3.2 Image Energy Functions**"节中也给出了一些其它的能量函数。而普林斯顿大学的作业[Programming Assignment 7: Seam Carving](http://www.cs.princeton.edu/courses/archive/fall14/cos226/assignments/seamCarving.html)中使用的是更简单的"*dual-gradient energy function*"。

###演示
待更新
