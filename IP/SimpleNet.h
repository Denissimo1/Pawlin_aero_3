#ifndef _SIMPLENET_H_
#define _SIMPLENET_H_
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cctype>       // std::toupper tolower
#include <iostream>

using namespace std;

class SimpleNetBasic {

public:
	vector<unsigned int> config;
	size_t nLayers;
	size_t nLayerSizeMax;
	size_t nWeights;
	size_t nBiases;
	size_t nInputs;
	size_t nOutputs;
	int output_activation;
	int activation_type;

	SimpleNetBasic():nLayers(0), nLayerSizeMax(0), nWeights(0),
		nBiases(0), nInputs(0), nOutputs(0), activation_type(0),  output_activation(0) {}
	SimpleNetBasic(const unsigned int *pConfig, size_t numLayers, int actType = 0, int outType = 0):nLayerSizeMax(0), nWeights(0),
		nInputs(0), nOutputs(0), activation_type(0), output_activation(0){
		nLayers = 0;
		setConfig(pConfig, numLayers, actType, outType);
	}
	size_t getStorageSize() const {
		return nLayerSizeMax + nLayerSizeMax;
	}
	void setConfig(const unsigned int* pConfig, size_t numLayers, int actType = 0, int outType = 0);
	inline void	 getConfig(vector<unsigned int> & config) const {
		if (nLayers != config.size()) throw("getConfig: unamtching config size");
		config = this->config;
	}
	inline const unsigned int*   getConfig() const { return (config.size() == 0) ? 0: &config[0]; }
	inline size_t  getMaxLayerSize()const { return nLayerSizeMax; }
	inline size_t  getTotalWeights() const { return nWeights + nBiases; }
	inline size_t  getWeightsCount() const { return nWeights; }
	inline size_t  getBiasesCount() const { return nBiases; }
	inline size_t  getInputsCount() const { return nInputs; }
	inline size_t  getOutputsCount() const { return nOutputs; }
	inline size_t  getLayersCount() const { return nLayers; }
	inline void	 setActivationType(int actType, int outType) {
		activation_type = actType;
		output_activation = outType;
	}
	int	getActivationType() const {
		return activation_type;
	}
	inline  int getOutputActivation() const {
		return output_activation;
	}
	void compute(const float* weights, const float* inputs, float* outputs) const;//compute one sample
	void compute(vector<float> &storage, const float* weights, const float* inputs, float* outputs) const;//compute one sample
	void save(FILE *file) const;
	void load(FILE *file);
	void saveBin(FILE* fp) const;
	void loadBin(FILE* fp);
	bool check_format(FILE* fp) const;
};
typedef SimpleNetBasic SimpleNetNoWeights;


class SimpleNet:public SimpleNetBasic {
public:
	enum  ActivationType {SIGMOID,  SIGMOID_SYMMETRIC, TANH, SIGMOID_SYMMETRIC2, LINEAR, HALFLINEAR, RELU, HALFLINEARPOS, RBF};
protected:
	vector<float> weights;
	
	vector<float> axons;
	

public:
	enum  ErrorType {NONE, MSE, MAXDIFF, ERROR1D};
	enum  Algorithm {RPROP, QUICKPROP, TEL, LM, MONTECARLO_MSE, MONTECARLO_MAX, ELM};
	enum  ResetType {RESET1, RESET2};
	float quality;

	SimpleNet():SimpleNetBasic() {}
	SimpleNet(const unsigned int *pConfig, size_t numLayers, ActivationType actType = SIGMOID, ActivationType outType = SIGMOID):SimpleNetBasic(),/*SimpleNetBasic(pConfig, numLayers,  actType, outType),*/ quality(0){
		nLayers = 0;
		setConfig(pConfig,numLayers, actType, outType); 
	}
	void allocateWeights();
	SimpleNet(SimpleNet &firstNet,SimpleNet &secondNet);
	void setConfig(const unsigned int* pConfig, size_t numLayers, ActivationType actType = SIGMOID, ActivationType outType = SIGMOID);

	const vector<float> & getWeights() const {return weights;}
	void setWeights(const vector<float> & weights){this->weights = weights;}
	inline float*  getWeightsArray(){return &weights[0];}
	inline const float*  getWeightsArray() const {return &weights[0];}
	

	void saveWeightsBin(FILE* fp) const;
	void loadWeightsBin(FILE* fp);

    void saveBin(FILE* fp) const;
	void loadBin(FILE* fp);

	void save(FILE *file) const;
	void load(FILE *file);
	void save(const char* filename, const char* mode = "wt" ) const;
	void load(const char* filename, const char* mode = "rt", fpos_t start_pos = fpos_t());
	void saveLogicalWeights(vector <vector <vector <float> > > &orderedweights) const;
	void printLogicalWeights() const;
	void loadLogicalWeights(const vector <vector <vector <float> > > &orderedweights);
	void setZeroWeights(){
		memset(&weights[0], 0, nWeights*sizeof(float));
	}
	void compute(const float* inputs, float* outputs) const;//compute one sample
	inline void compute(vector<float>& storage, const float * inputs, float * outputs) const
	{
		const float* pWeights = &weights[0];
		SimpleNetBasic::compute(storage, pWeights, inputs, outputs);
	}					
	void embeddNormalization(const float *shifts, const float *scales, size_t bufsize); //Embeds normalization into first neural layer weights
	void embeddOutputNormalization(const float* shifts, const float *scales_inv, size_t bufsize);

	~SimpleNet(){     }
	
	void printConfig(const char *name) const {
		printf("Network %s config:",name);
		for(size_t i = 0; i < config.size(); i++) printf("%s%d",i==0 ? "" : "-",config[i]);
		printf("\n");
	}
	void activate(float *array, size_t count) const;
	bool empty() const { return weights.empty(); }

};

class SimpleNetWithThreshold : public SimpleNet {
public:
	using SimpleNet::SimpleNet;
protected:
	float threshold = 0; // for decision models
public:
	void save(FILE *fp) const {
		SimpleNet::save(fp);
		fprintf(fp, "Threshold:%f\n", threshold);
	}
	void load(FILE *fp) {
		SimpleNet::load(fp);
		int count = fscanf(fp, "Threshold:%f\n", &threshold);
		if (count != 1) throw std::runtime_error("SimpleNetWithThreshold::load could not read threshold");
	}
	void load(const string &filename) {
		FILE *fp = fopen(filename.c_str(), "rt");
		load(fp);
		fclose(fp);
	}
	bool compute(const vector<float> &x, float &res) const {
		res = 0;
		SimpleNet::compute(&x[0], &res);
		return res > threshold;
	}
	float getThreshold() const {
		return threshold;
	}
	float &accessThreshold() {
		return threshold;
	}
};

#endif
