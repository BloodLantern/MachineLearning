# MachineLearning

This project uses submodules.
As such, when cloning you should use the `--recurse-submodule` flag.
If you already cloned the repository, instead run `git submodule update --init`.

This repository contains multiple projects that are categorized in Solution Folders in Visual Studio and Rider.

The `Common` folder contains library projects:
- `MachineLearning` is the library project containing all the code related to how the neural network works.
- `MonoGame.ImGuiNet` is an external library that allows the use of [ImGui](https://github.com/ocornut/imgui) in MonoGame projects.
- `MonoGame.Utils` contains common useful classes and functions for MonoGame projects.

The `Projects` folder contains the [MonoGame](https://monogame.net/) projects that use the previously mentioned libraries:
- `BackpropagationExamples` shows the usage of backpropagation on a neural network on three different examples:
  - A XOR gate
  - An AND gate
  - One in which you need to choose whether you are going to attack an enemy if you do or do not have life, ammo, and if the enemy is or isn't strong.
- `GeneticsExamples` shows the usage of genetics and natural selection on a population of neural networks on two examples:
  - A XOR gate
  - A simulation in which the neural networks control the rotation of an arrow that needs to move towards a random point on the window

Both projects make heavy use of ImGui windows to show the simulation state and settings.

The `GeneticsExamples` project currently doesn't work.
It seems the network learns at the beginning but stops after a certain number of iterations.
This is especially the case on the XOR gate example.
The network starts by learning the result of [0, 0], [0, 1] and [1, 0] inputs (tested with the Hyperbolic Tangent activation function) by then stops learning for the [1, 1] input and therefore is always wrong on it.
