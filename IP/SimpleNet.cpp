
#include "SimpleNet.h"
#include <math.h>
#include <fstream>


static const char *pwnlibheader = "(C) Pawlin Technologies Ltd FNN file format. PWNLIB 1.02";
void fGetLine(char * str, size_t strMaxLen, FILE * filePtr)
{
	fgets(str, (int)strMaxLen, filePtr);
	size_t headEndPtr = strlen(str) - 1;
	while (isspace(str[headEndPtr]))
	{
		str[headEndPtr--] = '\0';
	}
}

bool SimpleNetDescendingSort(const SimpleNet & net1, const SimpleNet & net2)
{
          return (net1.quality > net2.quality);
};

void SimpleNetBasic::setConfig(const unsigned int* pConfig, size_t numLayers, int actType, int outType) {
	if (1 < numLayers) {
		//if (config != 0)	{ delete[] config; config = 0;}

		//config = new unsigned int[numLayers];
		config.resize(numLayers);
		for (size_t i = 0; i< numLayers; i++)
			config[i] = pConfig[i];
		nLayers = numLayers;
		nInputs = config[0];
		nOutputs = config[nLayers - 1];
		nLayerSizeMax = nInputs;
		nWeights = 0;
		nBiases = 0;
		for (size_t i = 1; i< nLayers; i++) {
			nLayerSizeMax = max<size_t>(nLayerSizeMax, config[i]);
			nWeights += config[i - 1] * config[i];
			nBiases += config[i];
		}
		//weights.resize(nWeights + nBiases);
		//biases = &weights[0]+(nWeights-nBiases);
		//axons.resize(nLayerSizeMax + nLayerSizeMax);
	}
	activation_type = actType;
	output_activation = outType;
}
void SimpleNetBasic::save(FILE *file) const {
	fprintf(file, "%s\n", pwnlibheader);
	fprintf(file, "Layers:%zu\n", nLayers);
	fprintf(file, "Activations:%d\n", activation_type);
	fprintf(file, "Output activation:%d\n", output_activation);
	fprintf(file, "Config:");
	for (unsigned int i = 0; i < nLayers; i++) fprintf(file, "%d ", config[i]);
	/*fprintf(file, "\nWeights\n");
	//for(unsigned int i = 0; i < nWeights; i++) fprintf(file,"%.12f ", mpWeights[i]);

	const float* pWeights = &weights[0];
	const float* pBiases = pWeights + nWeights;
	for (unsigned int i = 1; i < nLayers; i++) {
		for (unsigned j = 0; j< config[i]; j++) {
			for (unsigned k = 0; k< config[i - 1]; k++) {
				fprintf(file, "%.12f ", *pWeights++);
			}
			fprintf(file, "%.12f ", *pBiases++);
		}
	}
	fprintf(file, "\n");*/
}

bool SimpleNetBasic::check_format(FILE* fp) const
{
	char buf[1024];
	fpos_t pos;
	fgetpos(fp, &pos);//save start pos
	fGetLine(buf, 1024, fp);
	bool res = !strcmp(buf, pwnlibheader);
	fsetpos(fp, &pos);//return start pos
	return res;
}
void SimpleNetBasic::load(FILE *file) {

	//printf("SimpleNet load.\n");
	char buf[1024];
	//fgets(buf,1024, file);
	fGetLine(buf, 1024, file);

	std::cout << "SimpleNetBasic::fGetLine done" << std::endl;

	if (strcmp(buf, pwnlibheader)) throw std::runtime_error("Wrong SimpleNet file format, no expected header.");

	std::cout << "SimpleNetBasic strcmp done" << std::endl;

	unsigned int layers = 0;
	fscanf(file, "Layers:%d\n", &layers);

	std::cout << "SimpleNetBasic fscanf layers done" << std::endl;

	fscanf(file, "Activations:%d\n", &activation_type);

	std::cout << "SimpleNetBasic fscanf activation_type done" << std::endl;

	fscanf(file, "Output activation:%d\n", &output_activation);

	std::cout << "SimpleNetBasic fscanfs output_activation done" << std::endl;

	if (layers == 0) throw("Incorrect number of layers");
	unsigned int *conf = new unsigned int[layers];
	memset(conf, 0, layers * sizeof(unsigned int));

	std::cout << "SimpleNetBasic memset done" << std::endl;

	fscanf(file, "Config:%d", conf + 0);
	
	std::cout << "SimpleNetBasic fscanfs Config done" << std::endl;
	//printf("have read layer config[%d]= %d \n",0,config[0],layers);
	for (unsigned int i = 1; i < layers; i++) {
		fscanf(file, " %d ", conf + i);
		//	printf("have read layer config[%d]= %d \n",i,config[i],layers);
	}

	std::cout << "SimpleNetBasic layers fscanf cycle done" << std::endl;

	setConfig(conf, layers, activation_type, output_activation);

	std::cout << "SimpleNetBasic setConfig done" << std::endl;
	
	delete[] conf;
	
	std::cout << "SimpleNetBasic delete[] conf done" << std::endl;
}

void SimpleNetBasic::saveBin(FILE* fp) const {
	fprintf(fp, "%s\n", pwnlibheader);
	fwrite(&nLayers, sizeof(unsigned int), 1, fp);//fprintf(file,"Layers:%d\n",nLayers);

	fwrite(&activation_type, sizeof(unsigned int), 1, fp);
	fwrite(&output_activation, sizeof(unsigned int), 1, fp);

	fwrite(&config[0], sizeof(unsigned int), nLayers, fp);

}
void SimpleNetBasic::loadBin(FILE* fp) {
	char buf[1024];
	fgets(buf, 1024, fp);
	if (strstr(buf, pwnlibheader) == NULL) {
		throw ("Wrong file format, no expected header.");
	}
	unsigned int layers = 0;
	fread(&layers, sizeof(unsigned int), 1, fp);//fscanf(file, "Layers:%d\n",&layers);
	fread(&activation_type, sizeof(unsigned int), 1, fp);//fscanf(file, "Activations:%d\n",&activation_type);
	fread(&output_activation, sizeof(unsigned int), 1, fp);//fscanf(file, "Activations:%d\n",&activation_type);

	if (layers == 0) throw("Incorrect number of layers");
	unsigned int *conf = new unsigned int[layers];
	memset(conf, 0, layers * sizeof(unsigned int));
	fread(&conf[0], sizeof(unsigned int), layers, fp);

	setConfig(conf, layers, activation_type, output_activation);
	delete[] conf;
}


void SimpleNet::allocateWeights() {
	weights.resize(nWeights + nBiases);
}
void SimpleNet::setConfig(const unsigned int* pConfig, size_t numLayers, ActivationType actType, ActivationType outType){

	SimpleNetBasic::setConfig(pConfig, numLayers,  actType,  outType);
	allocateWeights();
	
}



void SimpleNet::save(FILE *file) const{
	SimpleNetBasic::save(file);
	fprintf(file,"\nWeights\n");

	const float* pWeights = &weights[0];
	const float* pBiases = pWeights+nWeights;
	for(unsigned int i = 1; i < nLayers; i++){
		for (unsigned j = 0; j< config[i]; j++){
			for(unsigned k = 0; k< config[i-1]; k++){
				fprintf(file,"%.12f ", *pWeights++);
			}
			fprintf(file,"%.12f ", *pBiases++);
		}
	}
	fprintf(file,"\n");
}



void SimpleNet::load(FILE *file) {

	SimpleNetBasic::load(file);
	
	allocateWeights();
	
	char buf[1024];
	
	fGetLine(buf,1024, file);
	if(strcmp(buf,"Weights")) throw("Wrong file format, no expected weights section.");

	float* pWeights = &weights[0];
	float* pBiases = pWeights+nWeights;
	for(unsigned int i = 1; i < nLayers; i++){
		for (unsigned j = 0; j< config[i]; j++){
			for(unsigned k = 0; k< config[i-1]; k++) {
				//fprintf(file,"%.12f ", *pWeights++);
				fscanf(file," %f ",pWeights++);
			}
			fscanf(file," %f ",pBiases++);
			//fprintf(file,"%.12f ", *pBiases++);
		}
	}
}

void SimpleNet::save(const char* filename, const char* mode) const{
	FILE *f = fopen(filename,mode);
	if(f==NULL) throw("can't open file to save the network");
	save(f);

	fclose(f);
}
void SimpleNet::load(const char* filename, const char* mode, fpos_t start_pos){
	FILE *f = fopen(filename,mode);
	
	//FILE *f = fopen("re1.txt",mode);
	if (f == NULL)
	{
		//std::cout << "Simplenet can't open file to load the network " << string(filename) << "\n";
		throw std::runtime_error("SimpleNet::load can't open file to load the network");
	}
	//std::cout << "Trying to setpos in SimpleNet::load " << std::endl;
	fsetpos(f,&start_pos);
	//std::cout << "setpos in SimpleNet::load done " << std::endl;
	load(f);
	fclose(f);
}


void SimpleNet::saveWeightsBin(FILE* fp) const {
	//printf("Weights size:%d\n", weights.size());
	if (weights.size() > 0) {
		fwrite(&weights[0], sizeof(float), weights.size(), fp);
	}	
}
void SimpleNet::loadWeightsBin(FILE* fp) {
	allocateWeights();
	if (weights.size() > 0) {
		fread(&weights[0], sizeof(float), weights.size(), fp);
	}
}

void SimpleNet::saveBin(FILE* fp) const{

	SimpleNetBasic::saveBin(fp);  //save config
	saveWeightsBin(fp);//save weights

}
void SimpleNet::loadBin(FILE* fp){

	SimpleNetBasic::loadBin(fp);//load config
	loadWeightsBin(fp);//load weights

}
void SimpleNet::saveLogicalWeights(vector <vector <vector <float> > > &orderedweights) const{
	orderedweights.resize(nLayers-1);

	unsigned int inputs = config[0];//indexa;//mpConfig[0]
	const float* pWeights = &weights[0];
	const float* pBiases = pWeights+nWeights;
	for (unsigned int i = 1; i< nLayers; i++)
	{ 
		unsigned int neurons = config[i];
		orderedweights[i-1].resize(neurons);
		for (unsigned int j = 0; j< neurons; j++)
		{
			orderedweights[i-1][j].resize(inputs+1);
			for (unsigned int k = 0; k< inputs; k++) {
				orderedweights[i-1][j][k] = *pWeights++;//mpWeights[indexw++];
			}
			orderedweights[i-1][j][inputs] = *pBiases++;//mpWeights[indexw++];  // free weight of neuron
		}
		inputs = neurons;// number of inputs for next layer is the number of neurons from the previous
	}

}
void SimpleNet::printLogicalWeights() const{
	//int indexa = mpConfig[0];
	//int indexw = 0;
	unsigned int inputs = config[0];//indexa;//mpConfig[0]
	const float* pWeights = &weights[0];
	const float* pBiases = pWeights+nWeights;
	for (unsigned int i = 1; i< nLayers; i++)
	{ 
		unsigned int neurons = config[i];
		//orderedweights[i-1].resize(neurons);
		for (unsigned int j = 0; j< neurons; j++)
		{
			//orderedweights[i-1][j].resize(inputs+1);
			for (unsigned int k = 0; k< inputs; k++) {
				printf("%.2f ", *pWeights++);//mpWeights[indexw++]);
			}
			printf("%.2f\t  ", *pBiases++);// mpWeights[indexw++]);  // free weight of neuron
		}
		printf("\nNext layer: \n");
		inputs = neurons;// number of inputs for next layer is the number of neurons from the previous
	}
}
void SimpleNet::loadLogicalWeights(const vector <vector <vector <float> > > &orderedweights){
	if(orderedweights.size()!= nLayers-1) throw("Can't load from fewer layers");
	//printf("Loading weights\n");
	//unsigned indexa = (unsigned) mpConfig[0];
	//int indexw = 0;
	unsigned inputs = config[0];//indexa;//mpConfig[0]
	float* pWeights = &weights[0];
	float* pBiases = pWeights+nWeights;
	for (unsigned int i = 1; i< nLayers; i++)
	{ 
		//printf("Loading layer %d\n", i);
		unsigned neurons = config[i];
		for (unsigned j = 0; j< neurons; j++)
		{
			//printf("Loading neuron %d\n", j);
			//if(j>=orderedweights[i-1].size()) {indexw+=inputs+1; continue;} // no information about additional neurons
			if(j>=orderedweights[i-1].size()) {pWeights += inputs; pBiases++; continue;}
			for (unsigned k = 0; k< inputs; k++) {
				if(k>=(orderedweights[i-1][j].size()-1)) { pWeights++; continue;}

				//mpWeights[indexw++] = orderedweights[i-1][j][k];
				*pWeights++ = orderedweights[i-1][j][k];

				//printf("%f\t", orderedweights[i-1][j][k]);
			}
			
			//mpWeights[indexw++] = orderedweights[i-1][j].back();  // free weight of neuron
			*pBiases++ =  orderedweights[i-1][j].back(); 

			//printf("%f\n", orderedweights[i-1][j].back());
		}
		inputs = neurons;// number of inputs for next layer is the number of neurons from the previous
	}
	//printf("Loading weights done\n");
}

static void  batch_sigmoid(float*, int);
static void  batch_sigmoid_symm(float*, int);
static void  batch_tanh(float*, int);
static void  batch_sigmoid_symm2(float*, int);
static void  batch_linear(float*, int);
static void  batch_halflinear(float*, int);
static void  batch_relu(float*, int);
static void  batch_halflinearpos(float*, int);
static void  batch_rbf(float*, int);
static void (*batch_acivation[9])(float*, int) = {
	batch_sigmoid, 
	batch_sigmoid_symm,
	batch_tanh, 
	batch_sigmoid_symm2, 
	batch_linear,
	batch_halflinear,
	batch_relu,
	batch_halflinearpos,
	batch_rbf
};

inline float sigmfunc_safe(float x) {
	if (x < -10.0f) return 0.0f;
	if (x > +10.0f) return 1.0f;
	return  1.0f / (1.0f + expf(-x));
}

void  batch_sigmoid(float* x, int n){
	int i = 0;
	for (i  = 0; i< n-4; i+=4, x +=4){
	    float s0(0), s1(0), s2(0), s3(0);
		s0 = sigmfunc_safe(x[0]);
		s1 = sigmfunc_safe(x[1]);
		s2 = sigmfunc_safe(x[2]);
		s3 = sigmfunc_safe(x[3]);
		x[0] = s0; x[1] = s1;
		x[2] = s2; x[3] = s3;
	}
	for (; i< n; i++, x++){
	    x[0] = sigmfunc_safe(x[0]);
	}

}
void   batch_sigmoid_symm(float* x, int n){
	int i = 0;
	for (i  = 0; i< n-4; i+=4, x +=4){
	    float s0(0), s1(0), s2(0), s3(0);
		s0 = 2.0f/(1.0f+expf(-x[0])) - 1.0f;
		s1 = 2.0f/(1.0f+expf(-x[1])) - 1.0f;
		s2 = 2.0f/(1.0f+expf(-x[2])) - 1.0f;
		s3 = 2.0f/(1.0f+expf(-x[3])) - 1.0f;
		x[0] = s0; x[1] = s1;
		x[2] = s2; x[3] = s3;
	}
	for (; i< n; i++, x++){
		x[0] = 2.0f/(1.0f+expf(-x[0])) - 1.0f;
	}

}
void   batch_tanh(float* x, int n){
	int i = 0;
	for (i  = 0; i< n-4; i+=4, x +=4){
	    float s0(0), s1(0), s2(0), s3(0);
		s0 = 1.7159f*tanhf((2.0f/3.0f)*x[0]);
		s1 = 1.7159f*tanhf((2.0f/3.0f)*x[1]);
		s2 = 1.7159f*tanhf((2.0f/3.0f)*x[2]);
		s3 = 1.7159f*tanhf((2.0f/3.0f)*x[3]);
		x[0] = s0; x[1] = s1;
		x[2] = s2; x[3] = s3;
	}
	for (; i< n; i++, x++){
	    x[0]= 1.7159f*tanhf((2.0f/3.0f)* x[0]);
	}
}

void  batch_sigmoid_symm2(float* x, int n){
	int i = 0;
	for (i  = 0; i< n-4; i+=4, x +=4){
	    float s0(0), s1(0), s2(0), s3(0);
		s0 = sigmfunc_safe(x[0]) - 0.5f;
		s1 = sigmfunc_safe(x[1]) - 0.5f;
		s2 = sigmfunc_safe(x[2]) - 0.5f;
		s3 = sigmfunc_safe(x[3]) - 0.5f;
		x[0] = s0; x[1] = s1;
		x[2] = s2; x[3] = s3;
	}
	for (; i< n; i++, x++){
		x[0] = sigmfunc_safe(x[0]) - 0.5f;
	}
}
void   batch_linear(float*, int){
}
void   batch_halflinear(float* x, int n){
	int i = 0;
	for (i  = 0; i< n-4; i+=4, x +=4){
	    float s0(0), s1(0), s2(0), s3(0);
		s0 = x[0] < 0 ? 1.0f/(1.0f+expf(-x[0])) - 0.5f : x[0]*0.25f;
		s1 = x[1] < 0 ? 1.0f/(1.0f+expf(-x[1])) - 0.5f : x[1]*0.25f;
		s2 = x[2] < 0 ? 1.0f/(1.0f+expf(-x[2])) - 0.5f : x[2]*0.25f;
		s3 = x[3] < 0 ? 1.0f/(1.0f+expf(-x[3])) - 0.5f : x[3]*0.25f;
		x[0] = s0; x[1] = s1;
		x[2] = s2; x[3] = s3;
	}
	for (; i< n; i++, x++){
		x[0] = x[0] < 0 ? 1.0f/(1.0f+expf(- x[0])) - 0.5f : 0.25f*x[0];
	}
}

void   batch_halflinearpos(float* x, int n){
	int i = 0;
	for (i  = 0; i< n-4; i+=4, x +=4){
	    float s0(0), s1(0), s2(0), s3(0);
		s0 = x[0] < 0 ? 1.0f/(1.0f+expf(-x[0])) : x[0]*0.25f;
		s1 = x[1] < 0 ? 1.0f/(1.0f+expf(-x[1])) : x[1]*0.25f;
		s2 = x[2] < 0 ? 1.0f/(1.0f+expf(-x[2])) : x[2]*0.25f;
		s3 = x[3] < 0 ? 1.0f/(1.0f+expf(-x[3])) : x[3]*0.25f;
		x[0] = s0; x[1] = s1;
		x[2] = s2; x[3] = s3;
	}
	for (; i< n; i++, x++){
		x[0] = x[0] < 0 ? 1.0f/(1.0f+expf(- x[0])) : 0.25f*x[0]+0.5f;
	}
}

void   batch_relu(float* x, int n){
	int i = 0;
	for (i  = 0; i< n-4; i+=4, x +=4){
	    float s0(0), s1(0), s2(0), s3(0);
		s0 = x[0] < 0 ? 0 : x[0];
		s1 = x[1] < 0 ? 0 : x[1];
		s2 = x[2] < 0 ? 0 : x[2];
		s3 = x[3] < 0 ? 0 : x[3];
		x[0] = s0; x[1] = s1;
		x[2] = s2; x[3] = s3;
	}
	for (; i< n; i++, x++){
		x[0] = x[0] < 0 ? 0 : x[0];
	}
}

void   batch_rbf(float* x, int n){
	int i = 0;
	for (i  = 0; i< n-4; i+=4, x +=4){
	    float s0(0), s1(0), s2(0), s3(0);
		s0 = expf(-x[0]);
		s1 = expf(-x[1]);
		s2 = expf(-x[2]);
		s3 = expf(-x[3]);
		x[0] = s0; x[1] = s1;
		x[2] = s2; x[3] = s3;
	}
	for (; i< n; i++, x++){
		x[0] = expf(-x[0]);
	}

}
void computeOneLayer(const float* inputs, const float* const weights,const float* biases, float* outputs, 
					 unsigned int Ninputs, unsigned int Noutputs, int activation_type){
	
	const float* pWeights = weights;
	const float* pBiases =  biases;
	float* pOutputs = outputs;
	if (activation_type != SimpleNet::RBF){
		for (unsigned int i = 0; i< Noutputs; i++, pBiases++){
			const float* pInputs = inputs;
			float sum = 0.0f;
			for (unsigned int j = 0; j< Ninputs; j++, pWeights++, pInputs++){
				sum += pWeights[0] * pInputs[0];
			}
			if (activation_type == SimpleNet::LINEAR){
				*pOutputs++ = sum + pBiases[0]; // skribtsov, otherwise results differ from nnprc!!! (������� ����� ���� ��� ��� ((
			}
			else
				*pOutputs++ = sum + pBiases[0];
		}
	}
	else{

		for (unsigned int i = 0; i< Noutputs; i++){
			const float* pInputs = inputs;
			float sum = 0.0f;
			float c = 0.0f;//Kahan summation algorithm
			for (unsigned int j = 0; j< Ninputs; j++){
				float diff = (*pWeights++ - *pInputs++);
				float toadd = diff*diff - c;
				float t = sum + toadd;
				c = (t - sum) - toadd;
				sum = t;
			}
			sum *= *pBiases++;
			*pOutputs++ = sum;
		}

	}
	(*batch_acivation[activation_type])(outputs, (int)Noutputs);//batch activation

}
void SimpleNet::activate(float *array, size_t count) const {
	(*batch_acivation[activation_type])(array, (int)count);//batch activation

}
void SimpleNetBasic::compute(std::vector<float> &axons, const float* pWeights, const float* inputs, float* outputs) const {//compute one sample
	const float* pBiases = pWeights+nWeights;
	if (nLayers == 2){
		computeOneLayer(inputs,pWeights,pBiases, outputs, config[0], config[1], output_activation);
		return;
	}
	if (axons.size() < getStorageSize()) axons.resize(getStorageSize());
	float* pOprev = &axons[0];
	float* pOnext = pOprev + nLayerSizeMax;
	float* pO = pOprev;
	computeOneLayer(inputs,pWeights,pBiases, &axons[0], config[0], config[1], activation_type);//first layer
	pWeights += config[0]*config[1];
	pBiases += config[1];
	for (unsigned int j = 1; j < nLayers-2; j++)	{
		computeOneLayer(pOprev, pWeights,pBiases, pOnext, config[j], config[j+1], activation_type);//hidden layer
		pO = pOnext;
		pOnext = pOprev;
		pOprev = pO;
		pWeights += config[j]*config[j+1];
		pBiases += config[j+1];
	}
	computeOneLayer(pO,pWeights,pBiases, outputs, config[nLayers-2], config[nLayers-1], output_activation);//last layer
	
}
void SimpleNetBasic::compute(const float * weights, const float * inputs, float * outputs) const
{
	vector<float> storage;
	compute(storage, weights, inputs, outputs);
}
void SimpleNet::compute(const float* inputs, float* outputs) const {//compute one sample
	const float* pWeights = &weights[0];
	SimpleNetBasic::compute(pWeights,inputs,outputs);
}

void SimpleNet::embeddNormalization(const float *shifts, const float *scales, size_t bufsize){ //Embeds normalization into first neural layer weights
	//int indexw = 0;
	size_t inputs = (size_t)config[0];
	if(bufsize!=inputs) throw("Normalization embedding must be performed in physical network feature space\n");
	
	int neurons = config[1]; //first layer only
	float* pWeights = &weights[0];
	float* pBiases = pWeights+nWeights;
	for (int j = 0; j< neurons; j++) {
		float biasshift = 0;
		for (int k = 0; k< inputs; k++) {
			*pWeights *= scales[k];
			biasshift += *pWeights++ * shifts[k];
			//indexw++;
		}
		// bias
		//mpWeights[indexw] -= biasshift;		
		*pBiases++ -= biasshift;
		
		
		//indexw++;
	}
}

void SimpleNet::embeddOutputNormalization(const float* shifts, const float *scales_inv, size_t bufsize)
{
	vector<vector<vector<float> > > weights;
	saveLogicalWeights(weights);
	vector<vector<float> > &lastLayer = weights.back();
	if (lastLayer.size() != bufsize) throw std::runtime_error("SimpleNet::embeddOutputNormalization incompatible output layer neurons count != outnorm channels");
	for (size_t n = 0; n < lastLayer.size(); n++) {
		vector<float> &neuron = lastLayer[n];
		for (size_t i = 0; i < neuron.size(); i++) {
			if (i == neuron.size() - 1) {
				// bias
				neuron[i] = neuron[i] / scales_inv[n] + shifts[n];
			}
			else {
				neuron[i] /= scales_inv[n];
			}
		}
	}
	loadLogicalWeights(weights);
}

SimpleNet::SimpleNet(SimpleNet &firstNet,SimpleNet &secondNet){
	if(firstNet.getOutputsCount() != secondNet.getInputsCount()){
		throw std::runtime_error("SimpleNet::SimpleNet incompatible structure to join "
			+ std::to_string(firstNet.getOutputsCount()) + " != " +
			std::to_string(secondNet.getInputsCount())
		);
	}
	if (firstNet.output_activation != firstNet.activation_type) {
		throw std::runtime_error("SimpleNet::SimpleNet can't join non-homogenous network with other network\n");
	}
	if(firstNet.activation_type != secondNet.activation_type) {
		throw std::runtime_error("SimpleNet::SimpleNet can't join networks with different internal activation types\n");
	}

	quality=0;
	activation_type=firstNet.activation_type;
	output_activation= secondNet.output_activation;
	//state = 0;
	nLayers = firstNet.getLayersCount()+secondNet.getLayersCount()-1; //
	
	for(size_t k = 0; k < firstNet.getLayersCount(); k++){
		config.push_back(firstNet.getConfig()[k]);
	}
	
	for(size_t k = 1; k < secondNet.getLayersCount(); k++){ //first input=second output
		config.push_back(secondNet.getConfig()[k]);
	}
	//config
	nInputs = config[0];
	nOutputs = config[nLayers-1];
	nLayerSizeMax = nInputs;
	nWeights = 0;
	nBiases = 0;
	for (size_t i = 1; i< nLayers; i++){
		nLayerSizeMax = max<size_t>(nLayerSizeMax, config[i]);
		nWeights += config[i-1]*config[i];
		nBiases += config[i];
	}
	weights.resize(nWeights+nBiases);
	//biases = &weights[0]+(nWeights-nBiases);
	axons.resize(nLayerSizeMax+nLayerSizeMax);

	vector<vector<vector<float> > > firstNetWeights;
	firstNet.saveLogicalWeights(firstNetWeights);
	vector<vector<vector<float> > > secondNetWeights;
	secondNet.saveLogicalWeights(secondNetWeights);


	for(size_t layerCount = 0;layerCount < secondNetWeights.size();layerCount++){
		firstNetWeights.push_back(secondNetWeights[layerCount]);
	}
	loadLogicalWeights(firstNetWeights);
	printf("Create new net:\n");
	printConfig("Net:");

}