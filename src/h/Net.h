#ifndef NET_H_
#define NET_H_

#include "HiddenLayer.h";

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
};

} /* namespace nn */
#endif /* NET_H_ */
