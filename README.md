# Hybrid Quantum-CNN

**Quick setup:**
```bash
conda create -n qnn-env python=3.11 pytorch:pytorch pytorch:torchvision ipykernel 
conda activate qnn-env
pip install pennylane amazon-braket-pennylane-plugin dotenv matplotlib
```
Add your own AWS credentials in .env from example.env.txt

## Inspiration  
The idea started as a wild thought, blending quantum mechanics with neural networks. What began as a late-night “what if” moment turned into a deep dive into the frontier of quantum computing. We wanted to explore how the principles of superposition and entanglement could enhance the way neural networks learn. Many called it ambitious, maybe even crazy, but that challenge is exactly what inspired us to build it.


## What does it do?
It is a scaleable quantum neural network framework that trains convolutional network layers, which lead into quantum neural network layers. The quantum layers utilize efficient training based on a quantum mixture of experts (MoE) model that uses weighted summed nonunitary transformations. 
Some of the equations and theory we used:
Density QNN (quantum mixture) state: 

$$
\rho_D(\theta, \alpha, z) = \sum_{k=1}^{K} \alpha_k(z)\, U_k(\theta_k)\, \rho(z)\, U_k^\dagger(\theta_k)
$$


Data-dependent mixture weights (gating network), e.g. a softmax over a linear map of features: 

$$
α_k(z) = {softmax}_k(W z + b)
$$

Linear Combination of Unitaries (LCU) model (nonunitary weighted sum inside the channel): 


$$
f_LCU(θ, α, z) := Tr[ O (∑{k=1}^K α_k U_k(θ_k)) ρ(z) (∑{k=1}^K α_k U_k(θ_k))^\dagger ]
$$


## How we built it.

Our project compares classical image processing with quantum-inspired feature extraction. Each 4x4 patch of the image is passed through a quantum circuit that transforms the data into a new representation. This acts like a learned filter. We test two versions: one where the quantum features are computed during training, which is more flexible but very slow, and one where we precompute the quantum features once and train only the classical layers afterward. Precomputation drastically reduces training time because we avoid repeatedly simulating the quantum circuit. We then compare both quantum approaches to a standard CNN to evaluate speed and accuracy differences.

## The challenges we went through.
* Very complicated research. It took a while to understand and implement.
* Python Enviroments 
* Pennylane issues
* AWS braket issues
* Quantum Computers being offline for the weekend
* Pytorch compatibility
* Lack of Redbull
* Completely new approach
* Hard to debug Quantum Circuits

## Accomplishments we're proud of!
Got the CNN working well. A framework for the QNN was set up, and part of the research was put into code. Implemented entanglement ladder with reconfigurable beam splitter entangling gates. We were also proud of building a hybrid neural network where we run the quantum circuit once, essentially precomputing the quantum features, and using them for the classical neural network after. This proved to much faster on local machines.

## What did we learn?
We learned how to set up and use **API keys** to access quantum computers and delegate tasks to them. Building a **quantum neural network** turned out to be far more complex than a CNN. Involving qubits, quantum operators, and new types of layers. We also learned how to **optimize QNNs over time**, improving their performance through careful tuning and hybrid training strategies.  

## What's Next?
Although we didn’t get a fully functional QNN running at full scale, this project laid the groundwork for future exploration. Next steps include improving quantum simulation efficiency, refining the integration of quantum layers with classical networks, and experimenting with more advanced quantum operators. We plan to attempt to run the quantum feature extraction on IBM/IonQ/Rigetti devices to compare real-device noise vs. simulation and study performance differences. We also wanted to experiment with fewer gates, different structures of entanglement patterns, and newer hardware. We also plan to test hybrid approaches further and eventually run parts of the network on real quantum hardware to better understand their potential and limitations.


**Resources We Used for Research**
*https://engineering.lehigh.edu/sites/engineering.lehigh.edu/files/_DEPARTMENTS/ise/pdf/tech-papers/24/24T_011.pdf
*https://en.wikipedia.org/wiki/Quantum_neural_network
*https://pennylane.ai/qml/glossary/quantum_neural_network
*https://pennylane.ai/qml/demos/tutorial_quanvolution
*https://quantumzeitgeist.com/quantum-neural-variational-networks-models-forecast-multivariate-time-series-data-extending-complex-dependencies/
*https://www.nature.com/articles/s41534-025-01099-6
*https://scisimple.com/en/articles/2025-11-05-quantum-neural-networks-a-new-frontier-in-machine-learning--a98l0z8
*https://youtu.be/xL383DseSpE?si=lHwUaV7uTrddRB0r
