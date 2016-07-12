#ifndef NET_H_
#define NET_H_

#include <vector>
#include "HiddenLayer.h"

using namespace std;

namespace nn {

class Net {
public:
	Net();
	virtual ~Net();

	/* Layer handling */
	HiddenLayer newHiddenLayer();

	/* Neuron handling */
	InputNeuron newInputNeuron();
	OutputNeuron newOutputNeuron();
	BiasNeuron newBiasNeuron();
	BiasNeuron newBiasNeuron(HiddenLayer);
	ProcessingNeuron newProcessingNeuron();
	ProcessingNeuron newProcessingNeuron(HiddenLayer);

	/* Connection handling */
	bool connect(Neuron, Neuron, double);
	bool autoConnect();

	/* Using net */
	bool setInput(vector<double>);
	void processInput();
	bool processInput(vector<double>);
	vector<double> getOutput(); //

	/* Miscellaneous */
	InputLayer getInputLayer();
	OutputLayer getOutputLayer();
};

} /* namespace nn */
#endif /* NET_H_ */
