define({ entries : {
    "alaeddine2021deep": {
        "abstract": "Deep network in network (DNIN) model is an efficient instance and an important extension of the convolutional neural network (CNN) consisting of alternating convolutional layers and pooling layers. In this model, a multilayer perceptron (MLP), a nonlinear function, is exploited to replace the linear filter for convolution. Increasing the depth of DNIN can also help improve classification accuracy while its formation becomes more difficult, learning time gets slower, and accuracy becomes saturated and then degrades. This paper presents a new deep residual network in network (DrNIN) model that represents a deeper model of DNIN. This model represents an interesting architecture for on-chip implementations on FPGAs. In fact, it can be applied to a variety of image recognition applications. This model has a homogeneous and multilength architecture with the hyperparameter \u201cL\u201d (\u201cL\u201d defines the model length). In this paper, we will apply the residual learning framework to DNIN and we will explicitly reformulate convolutional layers as residual learning functions to solve the vanishing gradient problem and facilitate and speed up the learning process. We will provide a comprehensive study showing that DrNIN models can gain accuracy from a significantly increased depth. On the CIFAR-10 dataset, we evaluate the proposed models with a depth of up to L",
        "author": "Hmidi Alaeddine and Malek Jihene",
        "doi": "10.1155/2021/6659083",
        "journal": "Computational Intelligence and Neuroscience",
        "title": "Deep Residual Network in Network",
        "type": "article",
        "year": "2021"
    },
    "chaudhuri2000efficient": {
        "abstract": "In pattern recognition problems, the convergence of backpropagation training algorithm of a multilayer perceptron is slow if the concerned classes have complex decision boundary. To improve the performance, we propose a technique, which at first cleverly picks up samples near the decision boundary without actually knowing the position of decision boundary. To choose the training samples, a larger set of data with known class label is considered. For each datum, its k-neighbours are found. If the datum is near the decision boundary, then all of these k-neighbours would not come from the same class. A training set, generated using this idea, results in quick and better convergence of the training algorithm. To get more symmetric neighbours, the nearest centroid neighbourhood (Chaudhuri, Pattern Recognition Lett. 17 (1996) 11\u201317) is used. The performance of the technique has been tested on synthetic data as well as speech vowel data in two Indian languages.",
        "author": "B.B. Chaudhuri and U. Bhattacharya",
        "doi": "10.1016/S0925-2312(00)00305-2",
        "journal": "Neurocomputing",
        "pages": "11--27",
        "title": "Efficient training and improved performance of multilayer perceptron in pattern classification",
        "type": "article",
        "volume": "34",
        "year": "2000"
    },
    "golovko2000technique": {
        "abstract": "A new computational technique for training of multilayer feedforward neural networks with sigmoid activation function of the units is proposed. The proposed algorithm consists two phases. The first phase is an adaptive training step calculation, which implements the steepest descent method in the weight space. The second phase is estimation of calculated training step rate, which reaches a state of activity of the units for each training iteration. The simulation results are provided for the test example to demonstrate the efficiency of the proposed method, which solves the problem of training step choice in multilayer perceptrons.",
        "address": "Como, Italy",
        "author": "Vladimir Golovko and Yury Savitsky and T. Laopoulos and A. Sachenko and L. Grandinetti",
        "booktitle": "Proceedings of the IEEE-INNS-ENNS International Joint Conference on Neural Networks",
        "doi": "10.1109/IJCNN.2000.857856",
        "pages": "323--328",
        "title": "Technique of learning rate estimation for efficient training of MLP",
        "type": "inproceedings",
        "volume": "1",
        "year": "2000"
    },
    "ji2025comprehensive": {
        "abstract": "Through this comprehensive survey of Kolmogorov-Arnold Networks(KAN), we have gained a thorough understanding of its theoretical foundation, architectural design, application scenarios, and current research progress. KAN, with its unique architecture and flexible activation functions, excels in handling complex data patterns and nonlinear relationships, demonstrating wide-ranging application potential. While challenges remain, KAN is poised to pave the way for innovative solutions in various fields, potentially revolutionizing how we approach complex computational problems.",
        "author": "Tianrui Ji and Yuntian Hou and Di Zhang",
        "doi": "10.48550/arXiv.2407.11075",
        "howpublished": "\\url{https://doi.org/10.48550/arXiv.2407.11075}",
        "title": "A Comprehensive Survey on Kolmogorov Arnold Networks (KAN)",
        "type": "misc",
        "year": "2025"
    },
    "liu2024kan": {
        "abstract": "Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov-Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation functions on nodes (\"neurons\"), KANs have learnable activation functions on edges (\"weights\"). KANs have no linear weights at all -- every weight parameter is replaced by a univariate function parametrized as a spline. We show that this seemingly simple change makes KANs outperform MLPs in terms of accuracy and interpretability. For accuracy, much smaller KANs can achieve comparable or better accuracy than much larger MLPs in data fitting and PDE solving. Theoretically and empirically, KANs possess faster neural scaling laws than MLPs. For interpretability, KANs can be intuitively visualized and can easily interact with human users. Through two examples in mathematics and physics, KANs are shown to be useful collaborators helping scientists (re)discover mathematical and physical laws. In summary, KANs are promising alternatives for MLPs, opening opportunities for further improving today's deep learning models which rely heavily on MLPs.",
        "author": "Ziming Liu and Yixuan Wang and Sachin Vaidya and Fabian Ruehle and James Halverson and Marin Solja\u010di\u0107 and Thomas Y. Hou and Max Tegmark",
        "booktitle": "The Thirteenth International Conference on Learning Representations",
        "doi": "10.48550/arXiv.2404.19756",
        "title": "KAN: Kolmogorov--Arnold Networks",
        "type": "inproceedings",
        "year": "2025"
    },
    "qiu2025powermlp": {
        "abstract": "The Kolmogorov-Arnold Network (KAN) is a new network architecture known for its high accuracy in several tasks such as function fitting and PDE solving. The superior expressive capability of KAN arises from the Kolmogorov-Arnold representation theorem and learnable spline functions. However, the computation of spline functions involves multiple iterations, which renders KAN significantly slower than MLP, thereby increasing the cost associated with model training and deployment. The authors of KAN also noted that \"the biggest bottleneck of KANs lies in their slow training. KANs are usually 10x slower than MLPs, given the same number of parameters.\" To address this issue, we propose a novel MLP-type neural network PowerMLP that employs simpler non-iterative spline function representation, offering approximately the same training time as MLP while theoretically demonstrating stronger expressive power than KAN. Furthermore, we compare the FLOPs of KAN and PowerMLP, quantifying the faster computation speed of PowerMLP. Our comprehensive experiments demonstrate that PowerMLP generally achieves higher accuracy and a training speed about 40 times faster than KAN in various tasks.",
        "author": "Ruichen Qiu and Yibo Miao and Shiwen Wang and Yifan Zhu and Lijia Yu and Xiao-Shan Gao",
        "booktitle": "Proceedings of the AAAI Conference on Artificial Intelligence",
        "doi": "10.1609/aaai.v39i19.34210",
        "number": "19",
        "pages": "20069--20076",
        "title": "PowerMLP: An Efficient Version of KAN",
        "type": "inproceedings",
        "volume": "39",
        "year": "2025"
    },
    "wang2024lightweight": {
        "abstract": "Efficient and effective multivariate time series (MTS) forecasting is critical for real-world applications, such as traffic management and energy dispatching. Most of the current deep learning studies (e.g., Spatio-Temporal Graph Neural Networks and Transformers) fall short in a trade-off between performance and efficiency. Existing MTS forecasting studies have yet to fully and simultaneously address issues such as modelling both temporal and variate dependencies, as well as the temporal locality, hindering broader applications. In this paper, we propose a lightweight model, i.e., Time Series MLP (TSP). TSP is built upon MLP and relies on the PrecMLP with the proposed computationally free Precurrent mechanism to model both the variate dependency and temporal locality, thus being simple, effective and versatile. Extensive experiments show that TSP outperforms state-of-the-art methods on 16 datasets for both Long-term Time-series Forecasting and Traffic Forecasting tasks. Furthermore, it attains a significant reduction of at least 95.97% in practical training speed on the CPU.",
        "author": "Zhenghong Wang and Sijie Ruan and Tianqiang Huang and Haoyi Zhou and Shanghang Zhang and Yi Wang and Leye Wang and Zhou Huang and Yu Liu",
        "doi": "10.1016/j.knosys.2024.111463",
        "journal": "Knowledge-Based Systems",
        "pages": "111463",
        "title": "A lightweight multi-layer perceptron for efficient multivariate time series forecasting",
        "type": "article",
        "volume": "288",
        "year": "2024"
    },
    "xu2024kolmogorov": {
        "abstract": "Kolmogorov-Arnold Networks (KAN) is a groundbreaking model recently proposed by the MIT team, representing a revolutionary approach with the potential to be a game-changer in the field. This innovative concept has rapidly garnered worldwide interest within the AI community. Inspired by the Kolmogorov-Arnold representation theorem, KAN utilizes spline-parametrized univariate functions in place of traditional linear weights, enabling them to dynamically learn activation patterns and significantly enhancing interpretability. In this paper, we explore the application of KAN to time series forecasting and propose two variants: T-KAN and MT-KAN. T-KAN is designed to detect concept drift within time series and can explain the nonlinear relationships between predictions and previous time steps through symbolic regression, making it highly interpretable in dynamically changing environments. MT-KAN, on the other hand, improves predictive performance by effectively uncovering and leveraging the complex relationships among variables in multivariate time series. Experiments validate the effectiveness of these approaches, demonstrating that T-KAN and MT-KAN significantly outperform traditional methods in time series forecasting tasks, not only enhancing predictive accuracy but also improving model interpretability. This research opens new avenues for adaptive forecasting models, highlighting the potential of KAN as a powerful and interpretable tool in predictive analytics.",
        "author": "Kunpeng Xu and Lifei Chen and Shengrui Wang",
        "doi": "10.48550/arXiv.2406.02496",
        "howpublished": "\\url{https://doi.org/10.48550/arXiv.2406.02496}",
        "title": "Kolmogorov-Arnold Networks for Time Series: Bridging Predictive Power and Interpretability",
        "type": "misc",
        "year": "2024"
    },
    "yilmaz2022successfully": {
        "abstract": "The vanishing gradient problem (i.e., gradients prematurely becoming extremely small during training, thereby effectively preventing a network from learning) is a long-standing obstacle to the training of deep neural networks using sigmoid activation functions when using the standard back-propagation algorithm. In this paper, we found that an important contributor to the problem is weight initialization. We started by developing a simple theoretical model showing how the expected value of gradients is affected by the mean of the initial weights. We then developed a second theoretical model that allowed us to identify a sufficient condition for the vanishing gradient problem to occur. Using these theories we found that initial back-propagation gradients do not vanish if the mean of the initial weights is negative and inversely proportional to the number of neurons in a layer. Numerous experiments with networks with 10 and 15 hidden layers corroborated the theoretical predictions: If we initialized weights as indicated by the theory, the standard back-propagation algorithm was both highly successful and efficient at training deep neural networks using sigmoid activation functions.",
        "author": "Ahmet Yilmaz and Riccardo Poli",
        "doi": "10.1016/j.neunet.2022.05.030",
        "journal": "Neural Networks",
        "pages": "87--103",
        "title": "Successfully and efficiently training deep multi-layer perceptrons with logistic activation function simply requires initializing the weights with an appropriate negative mean",
        "type": "article",
        "volume": "153",
        "year": "2022"
    },
    "yu2025residual": {
        "abstract": "Despite their immense success, deep neural networks (CNNs) are costly to train, while modern architectures can retain hundreds of convolutional layers in network depth. Standard convolutional operations are fundamentally limited by their linear nature along with fixed activations, where multiple layers are needed to learn complex patterns, making this approach computationally inefficient and prone to optimization difficulties. As a result, we introduce RKAN (Residual Kolmogorov-Arnold Network), which could be easily implemented into stages of traditional networks, such as ResNet. The module also integrates polynomial feature transformation that provides the expressive power of many convolutional layers through learnable, non-linear feature refinement. Our proposed RKAN module offers consistent improvements over the base models on various well-known benchmark datasets, such as CIFAR-100, Food-101, and ImageNet.",
        "author": "Ray Congrui Yu and Sherry Wu and Jiang Gui",
        "doi": "10.48550/arXiv.2410.05500",
        "howpublished": "\\url{https://doi.org/10.48550/arXiv.2410.05500}",
        "note": "arXiv preprint arXiv:2410.05500",
        "title": "Residual Kolmogorov-Arnold Network for Enhanced Deep Learning",
        "type": "misc",
        "year": "2024"
    }
}});