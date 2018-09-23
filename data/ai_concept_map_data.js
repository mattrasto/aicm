const ai_concept_map_data = {
	"links": [
		{"source": "machine_learning", "target": "supervised_learning", "type": "derivative"},
		{"source": "machine_learning", "target": "reinforcement_learning", "type": "derivative"},
		{"source": "machine_learning", "target": "structured_prediction", "type": "derivative"},
		{"source": "machine_learning", "target": "anomaly_detection", "type": "derivative"},
		{"source": "machine_learning", "target": "dimen_reduction", "type": "derivative"},
		{"source": "machine_learning", "target": "unsupervised_learning", "type": "derivative"},
		{"source": "machine_learning", "target": "semi_supervised_learning", "type": "derivative"},
		{"source": "supervised_learning", "target": "svm", "type": "derivative"},
		{"source": "supervised_learning", "target": "neural_nets", "type": "derivative"},
		{"source": "supervised_learning", "target": "markov_model", "type": "derivative"},
		{"source": "supervised_learning", "target": "regressions", "type": "derivative"},
		{"source": "supervised_learning", "target": "naive_bayes", "type": "derivative"},
		{"source": "svm", "target": "nonlinear_svm", "type": "derivative"},
		{"source": "svm", "target": "linear_svm", "type": "derivative"},
		{"source": "svm", "target": "single_svm", "type": "derivative"},
		{"source": "neural_nets", "target": "convo_neural_nets", "type": "derivative"},
		{"source": "neural_nets", "target": "autoencoder", "type": "derivative"},
		{"source": "neural_nets", "target": "rec_neural_nets", "type": "derivative"},
		{"source": "neural_nets", "target": "gan", "type": "derivative"},
		{"source": "neural_nets", "target": "perceptron", "type": "derivative"},
		{"source": "neural_nets", "target": "stacked_autoencoders", "type": "derivative"},
		{"source": "neural_nets", "target": "self_organ_maps", "type": "derivative"},
		{"source": "neural_nets", "target": "adapt_reson_theory", "type": "derivative"},
		{"source": "neural_nets", "target": "replicator_nn", "type": "derivative"},
		{"source": "markov_model", "target": "markov_chains", "type": "derivative"},
		{"source": "markov_model", "target": "hidden_markov_model", "type": "derivative"},
		{"source": "regressions", "target": "log_regression", "type": "derivative"},
		{"source": "regressions", "target": "linear_regressions", "type": "derivative"},
		{"source": "linear_regressions", "target": "multi_linear_regression", "type": "derivative"},
		{"source": "linear_regressions", "target": "poly_regression", "type": "derivative"},
		{"source": "linear_regressions", "target": "simple_linear_regression", "type": "derivative"},
		{"source": "regressions", "target": "curvilinear_regression", "type": "derivative"},
		{"source": "structured_prediction", "target": "decision_trees_cart", "type": "derivative"},
		{"source": "structured_prediction", "target": "boost_algos", "type": "derivative"},
		{"source": "decision_trees_cart", "target": "classification_trees", "type": "derivative"},
		{"source": "decision_trees_cart", "target": "regression_trees", "type": "derivative"},
		{"source": "decision_trees_cart", "target": "ensemble_methods", "type": "derivative"},
		{"source": "decision_trees_cart", "target": "boosted_trees", "type": "derivative"},
		{"source": "boosted_trees", "target": "boost_algos", "type": "derivative"},
		{"source": "boot_aggregated", "target": "rand_forest", "type": "derivative"},
		{"source": "boost_algos", "target": "grad_boost", "type": "derivative"},
		{"source": "boost_algos", "target": "adaboost", "type": "derivative"},
		{"source": "anomaly_detection", "target": "ensemble_methods", "type": "derivative"},
		{"source": "anomaly_detection", "target": "static_rules", "type": "derivative"},
		{"source": "anomaly_detection", "target": "fuzzy_outlier_detection", "type": "derivative"},
		{"source": "anomaly_detection", "target": "cluster_analysis_outlier_detection", "type": "derivative"},
		{"source": "anomaly_detection", "target": "replicator_nn", "type": "derivative"},
		{"source": "anomaly_detection", "target": "single_svm", "type": "derivative"},
		{"source": "anomaly_detection", "target": "subspace_correlation", "type": "derivative"},
		{"source": "anomaly_detection", "target": "density_techniques", "type": "derivative"},
		{"source": "anomaly_detection", "target": "ensemble_methods", "type": "derivative"},
		{"source": "ensemble_methods", "target": "feature_bagging", "type": "derivative"},
		{"source": "ensemble_methods", "target": "score_norma", "type": "derivative"},
		{"source": "ensemble_methods", "target": "rotation_forest", "type": "derivative"},
		{"source": "ensemble_methods", "target": "boot_aggregated", "type": "derivative"},
		{"source": "density_techniques", "target": "local_outlier", "type": "derivative"},
		{"source": "density_techniques", "target": "knn", "type": "derivative"},
		{"source": "dimen_reduction", "target": "bayesian_models", "type": "derivative"},
		{"source": "dimen_reduction", "target": "missing_values", "type": "derivative"},
		{"source": "dimen_reduction", "target": "low_var_filter", "type": "derivative"},
		{"source": "dimen_reduction", "target": "multidimen_scaling", "type": "derivative"},
		{"source": "dimen_reduction", "target": "chisquare", "type": "derivative"},
		{"source": "dimen_reduction", "target": "stacked_autoencoders", "type": "derivative"},
		{"source": "dimen_reduction", "target": "decision_trees_ensembles", "type": "derivative"},
		{"source": "dimen_reduction", "target": "tsne", "type": "derivative"},
		{"source": "dimen_reduction", "target": "clustering", "type": "derivative"},
		{"source": "dimen_reduction", "target": "corr_analysis", "type": "derivative"},
		{"source": "dimen_reduction", "target": "rand_projections", "type": "derivative"},
		{"source": "dimen_reduction", "target": "pca", "type": "derivative"},
		{"source": "dimen_reduction", "target": "decision_trees_cart", "type": "derivative"},
		{"source": "dimen_reduction", "target": "forward_feat_selection", "type": "derivative"},
		{"source": "dimen_reduction", "target": "backward_feature", "type": "derivative"},
		{"source": "dimen_reduction", "target": "high_correlation", "type": "derivative"},
		{"source": "dimen_reduction", "target": "factor_analysis", "type": "derivative"},
		{"source": "dimen_reduction", "target": "nmf", "type": "derivative"},
		{"source": "pca", "target": "kernel_pca", "type": "derivative"},
		{"source": "pca", "target": "graph_kernel_pca", "type": "derivative"},
		{"source": "pca", "target": "blind_signal", "type": "derivative"},
		{"source": "factor_analysis", "target": "efa", "type": "derivative"},
		{"source": "factor_analysis", "target": "cfa", "type": "derivative"},
		{"source": "unsupervised_learning", "target": "knn", "type": "derivative"},
		{"source": "unsupervised_learning", "target": "clustering", "type": "derivative"},
		{"source": "unsupervised_learning", "target": "neural_nets", "type": "derivative"},
		{"source": "hierarch_clustering", "target": "agglomerative", "type": "derivative"},
		{"source": "hierarch_clustering", "target": "divisive", "type": "derivative"},
		{"source": "clustering", "target": "centroid_clustering", "type": "derivative"},
		{"source": "clustering", "target": "distri_clustering", "type": "derivative"},
		{"source": "clustering", "target": "hierarch_clustering", "type": "derivative"},
		{"source": "clustering", "target": "density_clustering", "type": "derivative"},
		{"source": "clustering", "target": "preclustering", "type": "derivative"},
		{"source": "clustering", "target": "corr_clustering", "type": "derivative"},
		{"source": "clustering", "target": "subspace_clustering", "type": "derivative"},
		{"source": "centroid_clustering", "target": "kmeans_clustering", "type": "derivative"},
		{"source": "centroid_clustering", "target": "kmedians_clustering", "type": "derivative"},
		{"source": "centroid_clustering", "target": "kmeans++_clustering", "type": "derivative"},
		{"source": "centroid_clustering", "target": "fuzzy_cmeans_clustering", "type": "derivative"},
		{"source": "distri_clustering", "target": "gauss_mixture", "type": "derivative"},
		{"source": "density_clustering", "target": "dbscan", "type": "derivative"},
		{"source": "density_clustering", "target": "optics", "type": "derivative"},
		{"source": "preclustering", "target": "canopy_clustering", "type": "derivative"},
		{"source": "corr_clustering", "target": "ccpivot", "type": "derivative"},
		{"source": "subspace_clustering", "target": "clique", "type": "derivative"},
		{"source": "subspace_clustering", "target": "subclu", "type": "derivative"},
		{"source": "latent_var_models", "target": "exp_max_algo", "type": "derivative"},
		{"source": "latent_var_models", "target": "meth_moments", "type": "derivative"},
		{"source": "blind_signal", "target": "latent_var_models", "type": "derivative"},
		{"source": "blind_signal", "target": "csp", "type": "derivative"},
		{"source": "blind_signal", "target": "ssa", "type": "derivative"},
		{"source": "blind_signal", "target": "lccad", "type": "derivative"},
		{"source": "blind_signal", "target": "nnmf", "type": "derivative"},
		{"source": "blind_signal", "target": "dca", "type": "derivative"},
		{"source": "blind_signal", "target": "ica", "type": "derivative"},
		{"source": "blind_signal", "target": "svd", "type": "derivative"},
		{"source": "semi_supervised_learning", "target": "graph_methods", "type": "derivative"},
		{"source": "semi_supervised_learning", "target": "generative_models", "type": "derivative"},
		{"source": "semi_supervised_learning", "target": "low_density_separation", "type": "derivative"},
		{"source": "semi_supervised_learning", "target": "gan", "type": "derivative"},
		{"source": "low_density_separation", "target": "transductive_svm", "type": "derivative"},
		{"source": "reinforcement_learning", "target": "evo_strategies", "type": "derivative"},
		{"source": "reinforcement_learning", "target": "markov_model", "type": "derivative"},
		{"source": "markov_model", "target": "markov_decision_processes", "type": "derivative"},
		{"source": "rec_neural_nets", "target": "clock_rnn", "type": "derivative"},
		{"source": "rec_neural_nets", "target": "gru", "type": "derivative"},
		{"source": "rec_neural_nets", "target": "neural_programmer", "type": "derivative"},
		{"source": "rec_neural_nets", "target": "diff_neural_comp", "type": "derivative"},
		{"source": "rec_neural_nets", "target": "neural_turing", "type": "derivative"},
		{"source": "rec_neural_nets", "target": "lstm", "type": "derivative"},
		{"source": "rec_neural_nets", "target": "act_rnn", "type": "derivative"}
	],
	"nodes": [
		{
			"id": "machine_learning",
			"name": "Machine Learning",
			"description": "Machine learning is the process of utilizing statistical models to learn from past data in order to provide clarity for new data. When doing machine learning, you need data. And often, you need a lot of data. Machine learning is deeply coupled with statistics, so to understand ML you need to understand what a statistical model does. Essentially, it takes past data and, based on the structure of that data, makes assumptions on similar data. For example, in predicting stock prices, the model may take in information such as the past prices, volumes, and other technical indicators. It uses this information to make guesses about the data. Depending on what you're trying to achieve, you can have the mode try to guess the future price, try to figure out whether some event caused a market panic, or find what kinds of data are related to each other. However, you have to be careful. As with anything in statistics, you have to be thorough in your methods. Results are not always as they seem!",
			"when": {
					"description": "These are general guidelines, but if you find any of these apply to your problem, machine learning may be helpful in finding a solution.",
					"cases": ["You have complex data", "You want to find patterns in your data", "You want to predict events", "You have lots of existing data", "You want to evolve your product to become better over time", "You've already sold your soul to statistics"]
				},
			"how": {
				"description": "Machine learning is done by optimizing a function that you specify to \"fit\" past data well, usually an error function. For example, you could say \"I want to predict house prices\" and train a model with historical data to try to make the predicted price as close as possible to the historical data's actual price. It does the hard work of finding the parameters that optimize the function for you! But be careful. Problems arise when the wrong model is chosen, bad data is used, the results are interpreted incorrectly, or the function is not fully optimized (this happens more often than you might think!).",
				"steps": ["Define your problem. Think of a very specific question you want to answer.", "Get the data relevant to finding that answer. Tons of it.", "Get more data.", "Clean your data. Figure out what parts are important. Umbrellas sold in NYC don't predict Moscow's daily temperature. (TODO: verify this)", "Decide which model / algorithm to use. Hint: this map will help!", "Pick your framework of choice.", "Build and train your model. If you do it right, you should have to wait a while for the training to finish.", "Test the results. Does it predict well? Is your data still nonsense? If not, use your knowledge gained and go back to step 1. Either way, you've now done machine learning!"]
			},
			"tools": {
				"description": "These are some of the most popular general machine learning tools.",
				"links": [
					{
						"name": "Scikit-Learn",
						"link": "http://scikit-learn.org/stable/",
						"description": "A popular open source Python ML library"
					},
					{
						"name": "mlpack",
						"link": "http://www.mlpack.org/",
						"description": "A C++ ML library focused on performance"
					},
					{
						"name": "Apache Spark MLLib",
						"link": "https://spark.apache.org/mllib/",
						"description": "Apache's ML library for Spark"
					},
					{
						"name": "Google Cloud ML",
						"link": "https://cloud.google.com/products/machine-learning/",
						"description": "Google's ML platform"
					},
					{
						"name": "Azure ML Studio",
						"link": "https://studio.azureml.net/",
						"description": "Microsoft's ML platform"
					},
					{
						"name": "Amazon ML",
						"link": "https://aws.amazon.com/machine-learning/",
						"description": "Amazon's ML platform"
					}
				]
			},
			"links": {
				"description": "Here are some of the best general machine learning tutorials I've come across.",
				"links": [
					{
						"name": "Coursera: Stanford Machine Learning",
						"link": "https://www.coursera.org/learn/machine-learning",
						"description": "This course by Andrew Ng is perfect for machine learning beginners. It covers the topics, math, and motivations for machine learning."
					},
					{
						"name": "TopTal Machine Learning Primer",
						"link": "https://www.toptal.com/machine-learning/machine-learning-theory-an-introductory-primer",
						"description": "This is a great introductory tutorial with an excellent example."
					},
					{
						"name": "Machine Learning Mastery",
						"link": "http://machinelearningmastery.com/start-here/#getstarted",
						"description": "This is one of my favorite blogs. This post is geared towards those just getting started."
					}
				]
			},
			"keywords": ["machine learning", "ai", "general", "predict", "classify", "reinforce", "improve", "data"]
		},
		{
			"id": "supervised_learning",
			"name": "Supervised Learning",
			"description": "Supervised learning is achieved by building a learning model and training the algorithm on labeled data points. Supervised learning can be broken down into two classes: prediction and classification. Both of these require that the algorithm knows something about what patterns the data holds so that it can predict or classify new examples properly. Therefore, we need what is called \"labeled\" data. This means that along with past examples' features, we also have what we as humans would consider the correct answer. Then, we feed these example + answer pairs into our algorithm to try to learn some way to represent this data in a way that it can then predict or classify new examples using this representation. This is called \"training\" the model. Supervised learning then, is any machine learning algorithm that uses labeled data to make guesses about new examples!",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "neural_nets",
			"name": "Neural Networks",
			"description": "Artificial neural networks are systems modeled after how our brain works; data is sent between neurons to come to a single conclusion. Neural networks are great for highly complex problems, such as image processing. They can also be leveraged to process traditionally difficult data, such as sequential data. It's important to note that \"neural network\" is an umbrella term, and that there are many different types of NNs with infinite ways to arrange them. Each neural network architecture, or topology, is engineered to work well for a specific type of data. For example, RNNs have a unique architecture that makes them very efficient at modeling sequential data.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "convo_neural_nets",
			"name": "CNN",
			"description": "A Convolutional Neural Network (CNN) is the favored type of model for image recognition. It essentially has two sections of layers: the first section contains convolutional and pooling layers that try to \"encode\" the input, and the second section that uses fully connected layers to try to learn a good representation of the encoded input. If this sounds convoluted, that's because it is.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "rec_neural_nets",
			"name": "RNN",
			"description": "Recurrent Neural Networks are designed to be useful for sequential data, making them very popular in natural language tasks, like NLP or handwriting / speech recognition. The way this is achieved is by adding feedback loops to all of the neurons in the hidden layers. This means the output of a neuron can feed back into itself. So how does this help us represent abritrary length sequences? Well, the feedback loop can be thought of as a way for the neuron to \"remember\" the data it processed in the past. If the neuron is trained to remember multiple states back, then a potentially infinite long sequence can be modeled! But this is hard. How does the neuron get trained in this way? We run into a problem where the neuron may forget past data, because it gets muddled with all of the previous data. After all, when it's training, it doesn't know whether the most important stage is 5 or 5,000 examples in the past. This is called the \"vanishing gradient\" problem. Despite this, RNNs are great for learning sequential data if they are engineered correctly. One thing to note is that the order in which the training data is fed into the model matters. Because RNNs look for sequential data, it wouldn't make sense to jumble up the sequences during training. For example, think about handwriting recognition; if you fed in the training data backwards (the reverse of your handwriting), it wouldn't be able to predict how you your normal writing!",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "lstm",
			"name": "LSTM",
			"description": "A long short-term memory network is an improved type of RNN that uses a \"memory cell\" in each neuron in its hidden layers to keep track of past information. Recall that RNNs are used primarily to model sequential data. One of the biggest problems with traditional RNNs is that long-term sequences are difficult to model. This is called the vanishing gradient problem, so named because the reason long sequences cannot be effectively modeled is that as we get further along in a sequence in the input data, weights will tend towards 0 and never cause the neuron's state to change. LSTMs solve this issue with allowing the cells to decide what information it holds onto over time. Each neuron's memory cell (usually) has three gates: input, forget, and output. The input gate allows new information to update the cell's memory, the forget gate determines which information the cell should throw away, and the output gate decides what information gets sent out from the current node. The neat thing about LSTMs is that these gates also learn what information to keep and let go over time! Not only does the LSTM configure its network weights dynamically, but it tries to remember the right information. This is extremely useful, and allows these types of networks to reach much better results for long-term sequential data.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": [
					{
						"name": "Understanding LSTM Networks",
						"link": "https://colah.github.io/posts/2015-08-Understanding-LSTMs/",
						"description": "Chris Olah provides a great explanation of LSTMs on his blog."
					}
				]
			},
			"keywords": []
		},
		{
			"id": "gru",
			"name": "GRU",
			"description": "A Gated Recurrent Unit is a derivative of the LSTM network model with performance improvements. While it uses the same idea of using a memory cell with gates to manage the flow of information in and out of each hidden layer neuron, except it only has two gates: reset and update. The reset gate determines how much information is allowed into the memory cell, and the update gate chooses how much memory needs to be retained.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "neural_turing",
			"name": "Neural Turing Machine",
			"description": "Neural Turing Machines are a type of RNN that allows every hidden layer neuron to access information from the same memory bank. The model is named for Alan Turing's computational model, which is to this day a critical piece of work in the computer science field.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "diff_neural_comp",
			"name": "Differentiable Neural Computer",
			"description": "A differentiable neural computer is an extension of neural turing machines, in that both have an external 'memory' that they are able to access, but it's able to store complex data structures as well. When trained, representations of more complex data structures begin to appear, allowing complex data to be modeled more efficiently than using normal external memory gates. Like earlier approaches, this memory can also be kept as long as needed, making DNCs well-suited for sequential data.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": [
					{
						"name": "DeepMind: Differentiable neural computers",
						"link": "https://deepmind.com/blog/differentiable-neural-computers/",
						"description": "DeepMind's blog covers their experience with DNCs."
					}
				]
			},
			"keywords": []
		},
		{
			"id": "clock_rnn",
			"name": "Clockwork RNN",
			"description": "The Clockwork RNN is an adaptation of basic RNN models that focuses on reducing model complexity and improving memory. The LSTM model was developed to improve memory in traditional RNNs for long sequences in data, but LSTM is computationally expensive. The clockwork RNN improves upon LSTM models by simplifying the memory architecture without affecting performance. In fact, as posited by the initial paper, clockwork RNNs score much better on time series prediction tasks than LSTMs. The clockwork RNN groups hidden layer neurons into \"modules\" that work at different \"speeds,\" which affect how fast the computations are performed and changes to the neuron state is propagated.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": [
					{
						"name": "Koutník et al.",
						"link": "https://arxiv.org/pdf/1402.3511v1.pdf",
						"description": "This paper from the folks at the Swiss AI lab IDSIA describes the method in full."
					}
				]
			},
			"keywords": []
		},
		{
			"id": "act_rnn",
			"name": "Adaptive Computation Time RNN",
			"description": "The ACT algorithm is a modification on traditional RNN architectures that allows the model to view each sample multiple times. This allows the neural net to learn how many times it needs to view each example to predict it correctly - instead of requiring a large amount of the same symbols over and over, ACT optimizes this by intelligently choosing which examples to review. When we study for exams, we don't repeat flashcards or problems we can solve with ease; similarly, ACT only repeats ones that it sruggles with identifying until the probability of correct classification is high enough.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "neural_programmer",
			"name": "Neural Programmer",
			"description": "A Neural Programmer is a neural network with some added built-in functionality, namely arithmetic and logic. Humans learn arithmetic and logic quickly, but neural networks don't have this innate ability. The Neural Programer determines when to use these operations by inferring them from the training samples, and can chain together these operations to achieve a high accuracy for some problems.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "gan",
			"name": "GAN",
			"description": "A Generative Adversarial Network (GAN) is a type of semi-supervised neural network that, in a very general way, attempts to perform a variation of the Turing Test to optimize itself. It does this by training two networks at the same time: a generator, and a discriminator.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "perceptron",
			"name": "FFNN / Perceptron",
			"description": "Perceptrons are an essential concept for neural networks. The earliest artificial neural networks were simply multi-layer perceptrons. While the more recently invented network topologies are used more commonly these days, there are still uses for simple feed-forward networks.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "autoencoder",
			"name": "Autoencoder",
			"description": "Autoencoders are a type of unsupervised neural network built for the purpose of simplifying an input into a more meaningful representation. The input layer is essentially \"compressed\" in the middle layers by forming a \"funnel\" with the network model; the hidden layers have fewer neurons than the input and output layers. The model is different than most neural networks, since its unsupervised. The goal of the network is to find a good representation of the input for other supervised methods, not do any predictions itself, so the output layer is actually the same size as the input layer. The network attempts to reconstruct the input layer in the output layer, but since the hidden layers have fewer neurons, some information is lost. This forces the model to only store the most essential attributes for representing the input in the hidden layers, and this more compact representation can be fed into other supervised algorithms to (hopefully) boost efficiency. The concept is similar to dimensionality reduction, where the input is reduced to ignore features that have offer little to the predictive power of the model. In the case of autoencoders, however, important aspects of a dimension may be kept, while part of the dimension may be discarded.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "stacked_autoencoders",
			"name": "Stacked Autoencoders",
			"description": "Some neural network architectures may benefit by using autoencoders in middle layers to reduce the 'representation' of the data into a compressed form. Stacked autoencoders are just this: neural networks with sparse autoencoder layers embedded as a hidden layer.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "svm",
			"name": "Support Vector Machines",
			"description": "SVMs are classifiers that distinguish between groups using a 'maximum-margin hyperplane,' which you can think of as a line separating examples from a dataset that is as wide as possible. By finding the widest line separating examples between the two classes, we assume this division is the best classification. SVMs are typically used as binary classifiers, though there are modified algorithms that can be used for multi-class classification. While typically SVMs are used for linear classification, you can map their inputs into high-dimensional feature spaces and then using a linear classifier in this high-dimensional feature space. This is called the 'kernel trick.'",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "nonlinear_svm",
			"name": "Non-Linear SVM",
			"description": "A non-linear SVM is simply an SVM that uses the kernel trick to classify datasets that aren't linearly separable. If you wish to use an SVM on a dataset best fit using, say, a quadratic function, then you'll need to use a proper kernel function (also known as a 'Mercer kernel') to map the data to a high-dimensional feature space in which you can linearly separate the data. Note that there are a limited number of kernel functions which can be used, as they must satisfy a few conditions in order to be viable. Most machine learning libraries support non-linear SVMs and provide these kernel functions for use.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "linear_svm",
			"name": "Linear SVM",
			"description": "A linear SVM is a traditional SVM that is able to classify datasets that are linearly separable. The SVM operates by finding two parallel planes (in 2D, these are lines) that successfully separate the data and calculating the distance between these two planes. The distance is called the margin, as it represents the margin between the two datasets. The SVM then attempts to find the two planes which result in the largest margin, indicating the largest distance between the datasets given any two planes.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "naive_bayes",
			"name": "Naive Bayes",
			"description": "Naive Bayes is a classifier based on statistics calculated on the existing dataset. It's friendly to high-dimensional data, and can be applied to both numeric and non-numeric data (though the latter is more common). Depending on the type of classifier used (there are typically three types: Gaussian, Bernoulli, and Multinomial), the exact statistics and formulas are used. In a Gaussian Naive Bayes classifier, the probabilities for a new sample are provided in terms of 'probability that new sample is class X' and 'ratio of all samples that are class X to all samples.' From this, we multiply both probabilities to find the 'prior probability' for each class, which is used to classify the new sample. Because samples are classified based on a combination of past classification ratios and an analysis of 'samples like it,' it is well-suited for realtime classification.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "regressions",
			"name": "Regressions",
			"description": "A regression is a measure of the relationship between a dependent variable (in machine learning, this is the class or Y) and one or more independent variables (the features). There are multiple ways to perform regression, but the goal is always to find some mathematical representation of this relationship. You can think of it as an 'approximation line' that could be straight, curved, or all manners of misshapen as long as it is mathematically representable.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "log_regression",
			"name": "Logistic Regression",
			"description": "A logistic regression is a type of regression that maps the input data to a certain class, typically 0 or 1. It uses the logit function, which outputs a number within [0, 1] that can be interpreted as the probability of that prediction being correct. You can use logistic regression with two classes, called binary logistic regression, three or more categories without ordering, called multinomial logistic regression, or three or more classes with ordering, called ordinal logistic regression.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "multi_linear_regression",
			"name": "Multiple Linear Regression",
			"description": "A multiple linear regression is a type of linear regression with multiple features. That's it - if you have more than one feature, then a linear regression is called a multiple linear regression. It can be used for prediction, but also for determining which features are most important. If your dataset is properly scaled to the number of features you have (typically recommended to be at least 10-20 times larger) then you can inspect the weights on each feature after training to answer the question 'what is the best predictor of Y?'",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "linear_regressions",
			"name": "Linear Regressions",
			"description": "A linear regression is a type of regression that maps the input data to a number. Linear regression is among the most widely used statistical techniques, as it's extremely simple but works in many scenarios. For example, in predicting the price of vegatables sold by the pound at your nearest market, a linear regression performs perfectly: there is one feature, vegetable weight, that has a linear (straight line) correlation with the price you'll pay.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "simple_linear_regression",
			"name": "Simple Linear Regression",
			"description": "A simple linear regression is a type of linear regression where there are only two variables: the independent variable (X) and the dependent variable (Y). It can be visualized as a 2D scatter plot where you attempt to fit the data by drawing a straight line through it.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "curvilinear_regression",
			"name": "Curvilinear Regression",
			"description": "A curvilinear regression is a type of regression that produces a curved line to fit the data. There are multiple equations for curved lines, such as exponential, power, logarithmic, trigonometric, and more. You can use similar, though slightly modified, equations to vanilla linear regression to fit these curved lines to the dataset.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "poly_regression",
			"name": "Polynomial Regression",
			"description": "Polynomial regression is a popular type of curivilinear regression. Polynomial regression is similar to multilinear regression except that instead of each weight being linear, they can also be exponential. In the case where only one independent variable is used, you simply add more weight terms increasing in power until you achieve satisfactory accuracy or give up. While you could theoretically form any curve from a polynomial regression, it's rare to find high-exponent weights, particularly for datasets with high dimensionality, since it becomes much more computationally expensive with every added feature or exponent.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "markov_chains",
			"name": "Markov Chains",
			"description": "Markov chains are a process which can be though of as a chain of states. Each state has a few arrows going to other states, each one with a probability that indicates how likely it is for that arrow to be crossed (that is, how likely it is for the process to move from the first state to the next one along that arrow). Markov chains, and in fact any Markov process, follow the Markov property, which states that the next state a process will enter *only* depends on the current state. While Markov chains themselves aren't particularly useful for machine learning, the Markov property is fundamental to some concepts within the space. Additionally, Markov chains are useful for modeling random processes and serve as an excellent tool for introductory modeling.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "markov_model",
			"name": "Markov Model",
			"description": "Markov models are stochastic models used to model randomly changing systems. Any system modeled by a Markov model follows the Markov property, which states that the next state a process will enter *only* depends on the current state. Markov models are used in some machine learning models where it's important to assume that only the current state is important in predicting the next state.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "hidden_markov_model",
			"name": "Hidden Markov Model",
			"description": "Hidden markov models are Markov chains in which the state is partially observable and thus can only be imprecisely determined. This is useful for some models, such as in speech signal classification where signals (voice data) are complex but can be decoded in a Markov chain by determining the most likely 'path' through the model in terms of a sequence of words.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "unsupervised_learning",
			"name": "Unsupervised Learning",
			"description": "Unsupervised learning is a field in machine learning where the data has no labels - only independent variables. Given a dataset with only descriptive variables, unsupervised models attempt infer patterns on this dataset - that is, infer the dependent variables with no prior examples of labels. A common task is clustering, which is used to partition the data into reasonable groups or classes.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "semi_supervised_learning",
			"name": "Semi-Supervised Learning",
			"description": "Semi-supervised learning is a field that utilizes models capable of combining both labeled and unlabeled data. Typically, these models are supervised models that perform well with unlabeled data. However, there are also some models that fall into neither supervised nor unsupervised learning that may be described as semi-supervised, such as generative adversarial models.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "reinforcement_learning",
			"name": "Reinforcement Learning",
			"description": "Reinforcement learning is a field with an entirely different problem statement than supervised and unsupervised methods. In reinforcement learning, you have agents, actions, an environment, and a reward function. Agents live in an environment (state space) that can perform a certain set of actions and must use these actions to progress to a state in their environment that maximizes their cumulative reward. For example, in video games you have a character (agent), some controls (actions), a level or playing field (environment), and a reward function (points, gold, or a goal: win).",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "anomaly_detection",
			"name": "Anomaly Detection",
			"description": "Anomaly detection is the field focused on detecting outliers in a dataset. When data has a certain pattern to it, such as bank users making payments on a regular basis, outliers can be detected by identifying abnormal behavior in the dataset.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "dimen_reduction",
			"name": "Dimensionality Reduction",
			"description": "Machine learning constantly deals with very high-dimensional data, sometimes having tens of thousands (or more!) dimensions. While having large amounts of multi-faceted data is helpful, it also has its drawbacks. The \"curse of dimensionality\" is a real problem in the field, describing the issues that start to arise when using machine learning to wield high-dimensional data. To combat these problems, there are methods to reduce the dimensionality of the data. In the same way that we can compress a file to save space, we can compress our data to save statistical power. However, we can't \"compress\" our data too much, or we may start to lose important information regarding our dataset, damaging our ability to obtain accurate results. There exist many algorithms for reducing the dimensionality of data, each with their own spot on the balance between compression and relevant information retention.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "structured_prediction",
			"name": "Structured Prediction",
			"description": "Structured prediction, or structured prediction, is a general term for classification or regression with an interpretable structure or framework. One of the complaints of machine learning, specifically with neural networks, is that we 'don't know what they're doing.' Structured learning is accomplished by using models that tell us what they're doing, such as decision trees. If a classification can be made by saying 'if this variable is less than 10, this sample is this class, otherwise it's the other class,' then structured prediction may not only be viable, but more helpful for researchers to use and understand.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "decision_trees_cart",
			"name": "Decision Trees (CART)",
			"description": "Decision trees are used to organize your thoughts based on the decisions you make. For example, your tree can help guide your decision process in deciding what kind of clothes to wear in the morning. If it's sunny, you might wear sunglasses. We can apply this structure to classic regressions by forming two kinds of trees: classification and regression trees. In classification trees, leaf nodes are class labels and each branch is a linear combination of some features. For example, leaf A might represent \"I will wear jeans today\" while leaf B represents \"I will wear shorts today.\" Only one of these decisions can be made, making this a classification problem. Regression trees are trees where the leaves are calculated with weights. Using the same labels as the last example, leaf A might have a value of 100 while leaf B has a value of 20, and this can be considered a \"decision\" weighted more heavily towards leaf A; that is, you are more likely to wear jeans today than shorts. Decision trees are a great tool for visualizing these processes, as its structure lends is very intuitive to humans.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "classification_trees",
			"name": "Classification Trees",
			"description": "Classification Trees are a type of decision tree that can be thought of as \"making a decision\" to classify something. Let's say you wanted to determine whether an image was of a cat or dog. As a human, we look for distinctive features like snouts, ear shape, size, color, etc. Classification trees do the same thing, but are formed via statistical methods. It will try to find its own representation of the possible classes by making \"decisions\" on the classes you feed it. Remember that leaf nodes in a decision tree indicate the \"output\" of the algorithm. That means that in classification trees, the leaf nodes are classes. Internal nodes are conditions like \"is there a snout\", and as we move down the tree we may even be able to understand the \"state\" of the node as a more general understanding of the classes! For example, if we fed our cat vs. dog classificatier a picture of a car, it might create a tree with the first decision as \"does it have wheels?\" Then, whichever nodes are linked to from the \"No\" branch of this decision may indicate \"animals\" as a more general class!",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "regression_trees",
			"name": "Regression Trees",
			"description": "Regression trees are decision trees that can take and operate on continuous dependent variables. In cases where the output of the decision tree should be a continuous variable instead of a class, you'll want to use regression trees.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "ensemble_methods",
			"name": "Ensemble Methods",
			"description": "Ensemble methods are techniques that generate multiple independent models and combine them after they are individually trained to improve performance. Ensembles are especially useful for models prone to overfitting or focusing on outliers, as it aggregates the models. This reduces reliance on any particular pattern unless it's relevant across a large subset of the individual models, which is likely what we'd hope to occur.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "boosted_trees",
			"name": "Boosted Trees",
			"description": "Gradient boosted trees are decision trees or decision forests to which gradient boosting is applied. Gradient boosting can be thought of as an optimization of the cost function such that gradients are more likely to point in the negative direction. When applied to individual decision trees, the quality of fit of each model is improved. This causes boosted trees to perform similar to decision forests, in that abnormalities are smoothed out; with decision forests, this is done through aggregation, while for boosted trees this is done on the individual model.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "rotation_forest",
			"name": "Rotation Forest",
			"description": "Rotation forests are a special type of decision forest intended to better represent feature diversity. Before training, the data is separated into subsets and Principal Component Analysis (PCA) is applied to each subset. The principal components are then used to train a tree with the entire dataset. Once all decision trees have been trained with the whole dataset (using only their respective principal components), they are aggregated.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "boot_aggregated",
			"name": "Bootstrap Aggregated",
			"description": "Bootstrap Aggregating, or bagging, is another type of aggregation method commonly appled to decision trees (though it can also be applied to any method). Instead of training multiple decision trees with the entire dataset, as with vanilla decision forests, only a subset of the data is used to train each base classifier (tree). They are then combined to form a forest. The goal of bagging is to better represent feature diversity by randomly sampling the dataset.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "rand_forest",
			"name": "Random Forest",
			"description": "A random forest is an aggregation method by which decision trees are combined into a single classifier. A single decision tree may be prone to overfitting or being weighted towards certain data abnormalities, but a random forest mitigates these issues by combining multiple classifiers in the hope that the ensemble will better represent the data's true nature and \"smooth\" out the abnormalities.",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "boost_algos",
			"name": "Boosting Algorithms",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "grad_boost",
			"name": "Gradient Boosting",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "adaboost",
			"name": "AdaBoost",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "feature_bagging",
			"name": "Feature Bagging",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "score_norma",
			"name": "Score Normalization",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "static_rules",
			"name": "Static Rules",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "fuzzy_outlier_detection",
			"name": "Fuzzy-Logic-Based Outlier Detection",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "cluster_analysis_outlier_detection",
			"name": "Cluster-Analysis-Based Outlier Detection",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "replicator_nn",
			"name": "Replicator NN",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "single_svm",
			"name": "Single Class SVM",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "subspace_correlation",
			"name": "Subspace-Based / Correlation-Based",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "density_techniques",
			"name": "Density-Based Techniques",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "knn",
			"name": "K-Nearest Neighbor",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "local_outlier",
			"name": "Local Outlier Factor",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "high_correlation",
			"name": "High Correlation",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "backward_feature",
			"name": "Backward Feature Elimination",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "forward_feat_selection",
			"name": "Forward Feature Selection / Construction",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "nmf",
			"name": "NMF",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "pca",
			"name": "Principal Component Analysis",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "graph_kernel_pca",
			"name": "Graph-Based Kernel PCA",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "kernel_pca",
			"name": "Kernel PCA",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "rand_projections",
			"name": "Random Projections",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "corr_analysis",
			"name": "Correspondence Analysis",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "clustering",
			"name": "Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "tsne",
			"name": "t-SNE",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "decision_trees_ensembles",
			"name": "Decision Tree Ensembles",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "chisquare",
			"name": "Chi-square / Information Gain",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "multidimen_scaling",
			"name": "Multidimensional Scaling",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "low_var_filter",
			"name": "Low Variance Filter",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "missing_values",
			"name": "Missing Values Ratio",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "bayesian_models",
			"name": "Bayesian Models",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "factor_analysis",
			"name": "Factor Analysis",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "efa",
			"name": "Exploratory Factor Analysis",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "cfa",
			"name": "Confirmatory Factor Analysis",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "latent_var_models",
			"name": "Latent Variable Models",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "blind_signal",
			"name": "Blind Signal Separation",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "svd",
			"name": "SVD",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "ica",
			"name": "ICA",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "dca",
			"name": "DCA",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "nnmf",
			"name": "NNMF",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "lccad",
			"name": "LCCAD",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "ssa",
			"name": "SSA",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "csp",
			"name": "CSP",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "meth_moments",
			"name": "Method of Moments",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "exp_max_algo",
			"name": "Expectation-Maximization Algorithm",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "self_organ_maps",
			"name": "Self-Organizing Maps",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "adapt_reson_theory",
			"name": "Adaptive Resonance Theory",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "hierarch_clustering",
			"name": "Hierarchical Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "agglomerative",
			"name": "Agglomerative",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "divisive",
			"name": "Divisive",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "centroid_clustering",
			"name": "Centroid-Based Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "kmeans_clustering",
			"name": "k-means Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "kmedians_clustering",
			"name": "k-medians Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "kmeans++_clustering",
			"name": "k-means++ Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "fuzzy_cmeans_clustering",
			"name": "Fuzzy c-means Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "distri_clustering",
			"name": "Distribution-Based Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "gauss_mixture",
			"name": "Gaussian Mixture Models",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "density_clustering",
			"name": "Density-Based Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "dbscan",
			"name": "DBSCAN",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "optics",
			"name": "OPTICS",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "preclustering",
			"name": "Pre-Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "canopy_clustering",
			"name": "Canopy Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "corr_clustering",
			"name": "Correlation Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "ccpivot",
			"name": "CC-Pivot",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "subspace_clustering",
			"name": "Subspace Clustering",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "clique",
			"name": "CLIQUE",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "subclu",
			"name": "SUBCLU",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "graph_methods",
			"name": "Graph-Based Methods",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "generative_models",
			"name": "Generative Models",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "low_density_separation",
			"name": "Low-Density Separation",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "transductive_svm",
			"name": "Transductive SVM",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "evo_strategies",
			"name": "Evolution Strategies",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		},
		{
			"id": "markov_decision_processes",
			"name": "Markov Decision Processes",
			"description": "",
			"when": {
					"description": "",
					"cases": []
			},
			"how": {
				"description": "",
				"steps": []
			},
			"tools": {
				"description": "",
				"links": []
			},
			"links": {
				"description": "",
				"links": []
			},
			"keywords": []
		}
	]
}
