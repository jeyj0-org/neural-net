#include "cpp/Net.cpp";

int main() {
	nn::testInterface();
	return 0;
}

namespace nn {

void testInterface() {
	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// Creating a neural-net

	/*
	 * Creating a new empty net does not require any of the containing
	 * objects to be created beforehand.
	 *
	 * A newly created net already contains an input-layer and an
	 * output-layer. Every other layer is a new processing layer in-between
	 * the two.
	 */
	Net net = new Net;

	/*
	 * This creates a new input-neuron on the input-layer. It is not yet
	 * possible to create an input-neuron on another layer, but there should
	 * not be the need for that either.
	 */
	InputNeuron inputNeuron = net->newInputNeuron();

	/*
	 * This creates a new output-neuron on the output-layer. It is not yet
	 * possible to create an output-neuron on another layer, but there
	 * should not be the need for that either.
	 */
	OutputNeuron outputNeuron = net->newOutputNeuron();

	/*
	 * This creates a new processing layer.
	 */
	HiddenLayer hiddenLayer = net->newHiddenLayer();

	/*
	 * This creates a new bias-neuron on the last added layer. This includes
	 * the input layer and excludes the output layer. So if this is called
	 * right after net-creation, the bias neuron will be added to the
	 * input-layer.
	 */
	BiasNeuron biasNeuron1 = net->newBiasNeuron();

	/*
	 * This creates a new bias-neuron on the given layer. This layer can be
	 * any hidden-layer, or the input layer, but not the output-layer.
	 */
	BiasNeuron biasNeuron2 = net->newBiasNeuron(hiddenLayer);

	/*
	 * This creates a new processing-neuron on the last added processing
	 * layer.
	 */
	ProcessingNeuron processingNeuron1 = net->newProcessingNeuron();

	/*
	 * This creates a new processing-neuron on the layer that was given as
	 * argument. It is not possible to use the input- or output-layer for
	 * this.
	 */
	ProcessingNeuron processingNeuron2 = net->newProcessingNeuron(hiddenLayer);

	/*
	 * With this function neurons can be connected.
	 *
	 * net.connect(Neuron FROM, Neuron TO, double WEIGHT);
	 */
	bool connection1 = net->connect(inputNeuron, processingNeuron1, 1D);
	bool connection2 = net->connect(inputNeuron, processingNeuron2, 0.5D);

	/*
	 * With this functions neurons can be automatically connected. This
	 * means that every neuron will be connected to every neuron of the
	 * previous layer, all with the same weights. This is the simplest way
	 * to connect simple nets that are not optimized yet and should still be
	 * trained.
	 *
	 * Notice that this will also reset all previous connections.
	 */
	bool connections = net->autoConnect();

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// Using the net

	/*
	 * Every net has input-neurons. These input-neurons can measure 0-100%.
	 * So for every processing the net needs an input on each input-neuron.
	 * To realize this, the net needs to get the inputs as an double[]. The
	 * most left and top neuron will get double[0] as input-value. Then the
	 * most left, but second-from-top neuron will get double[1] as
	 * input-value. And so on, until the left column is completely finished.
	 * Normally there only is one, but maybe someone created a special net,
	 * so then the next column will be worked through from top to bottom,
	 * and so on. So the double[] given has to have the same amount of
	 * values as there are input-neurons in the net.
	 */
	double values[] = new double[] {1D};
	bool success1 = net->setInput(values);

	/*
	 * After all the input is set for the next run, the net has to process
	 * that input. This has to be initialized by a function so that
	 * simulations can adopt this better.
	 */
	net->processInput();

	/*
	 * Combination of the two methods above:
	 *
	 * net.setInput(values);
	 *
	 * net.processInput();
	 */
	bool success2 = net->processInput(values);

	/*
	 * Finally the values of the output-neurons can be read. This way the
	 * output can be analyzed and actions can be taken depending on it. The
	 * double[] will be in the same format as the input-double[] was too.
	 * Only that the length will be the same amount as output-neurons, not
	 * input-neurons, obviously.
	 */
	double output[] = net->getOutput();

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// Miscellaneous functions to use on the net and its components

	/*
	 * With this method the input-layer can be retrieved from the net.
	 */
	InputLayer inputLayer = net->getInputLayer();

	/*
	 * With this method the output-layer can be retrieved from the net.
	 */
	HiddenLayer outputLayer = net->getOutputLayer();
}

} /* namespace nn */
