package weka.classifiers.bayes;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;
import java.lang.Math;
import java.util.Random;
import java.util.Collections;

import weka.classifiers.*;
import weka.classifiers.RandomizableClassifier;
import weka.clusterers.SimpleKMeans;
import weka.core.*;
import weka.core.Capabilities.Capability;

import weka.core.matrix.DoubleVector;
import weka.core.matrix.Matrix;
import weka.estimators.DiscreteHMMEstimator;
import weka.estimators.HMMEstimator;
import weka.estimators.MultivariateNormalEstimator;
import weka.estimators.MultivariateNormalHMMEstimator;

/*
 * A hidden Markov Model classifier class.
 */
public class HMM extends weka.classifiers.RandomizableClassifier implements weka.core.OptionHandler, weka.core.MultiInstanceCapabilitiesHandler {
	
	private static final long serialVersionUID = 1959669739718119361L;

	protected class ProbabilityTooSmallException extends Exception
	{
		private static final long serialVersionUID = -2706223192260478060L;

		ProbabilityTooSmallException(String s)
		{
			super(s);
		}
	}
	
	/** The number of classes */
	protected int m_NumStates = 6;
	protected int m_NumOutputs;
	protected int m_OutputDimension = 1;
	protected boolean m_Numeric = false;
	protected double m_IterationCutoff = 0.01;
	protected int m_SeqAttr = -1;
	
	/*
	 * gets the index of the data attribute containing the sequence data
	 */
	public int getSequenceAttribute()
	{
		return m_SeqAttr;
	}
	
	protected Random m_rand = null;
	
	protected double minScale = 1.0E-200;

	protected boolean m_RandomStateInitializers = false;
	
	/**
	 * gets whether the HMM state probabilities are randomly initialized.
	 * If this is false (default) the states are set using clustering on the 
	 * dataset. 
	 */
	public boolean isRandomStateInitializers() {
		return m_RandomStateInitializers;
	}

	/**
	 * sets whether the HMM state probabilities are randomly initialized.
	 * If this is false (default) the states are set using clustering on the 
	 * dataset. 
	 * 
	 * @param randomStateInitializers if true the HMM state probabilities will be initialized randomly, if false they will be initialized using clustering
	 */
	public void setRandomStateInitializers(boolean randomStateInitializers) {
		this.m_RandomStateInitializers = randomStateInitializers;
	}

	protected boolean m_Tied = false;

	/**
	 * gets whether the covariance matrices of guassian HMMs are tied. 
	 * Tied covariances are the same for all states of the HMM. 
	 * Using tied covariances makes it possible to learn models from less
	 * data but restricts the class of models that can be learned.
	 */
	public boolean isTied() {
		return m_Tied;
	}

	/**
	 * sets whether the covariance matrices of guassian HMMs are tied. 
	 * Tied covariances are the same for all states of the HMM. 
	 * Using tied covariances makes it possible to learn models from less
	 * data but restricts the class of models that can be learned.
	 * 
	 * @param tied if true the covariances will be tied
	 */
	public void setTied(boolean tied) {
		m_Tied = tied;
	}
	
	/**
	 * the type of covariance matrices for gaussian HMMs.
	 * A FULL matrix allows arbitrary dependencies between variables, 
	 * a DIAGONAL matrix assumes that all variables are independent. 
	 * SPHERICAL models assume that variables are independent and all have 
	 * the same variance. SPHERICAL covariances imply a more restricted 
	 * set of models than DIAGONAL covariances which are more restricted 
	 * than FULL matrices. Restricted models require less data to learn
	 * but may not be able to model all features of the data. 
	 */
	public static final Tag [] TAGS_COVARIANCE_TYPE = {
	    new Tag(MultivariateNormalEstimator.COVARIANCE_FULL, "Full matrix (unconstrianed)"),
	    new Tag(MultivariateNormalEstimator.COVARIANCE_DIAGONAL, "Diagonal matrix (no correlation between data attributes)"),
	    new Tag(MultivariateNormalEstimator.COVARIANCE_SPHERICAL, "Spherical matrix (all attributes have the same variance)"),
	};
	
	protected int m_CovarianceType = MultivariateNormalEstimator.COVARIANCE_FULL;
	
	/**
	 * get the type of covariance matrices for gaussian HMMs.
	 * A FULL matrix allows arbitrary dependencies between variables, 
	 * a DIAGONAL matrix assumes that all variables are independent. 
	 * SPHERICAL models assume that variables are independent and all have 
	 * the same variance. SPHERICAL covariances imply a more restricted 
	 * set of models than DIAGONAL covariances which are more restricted 
	 * than FULL matrices. Restricted models require less data to learn
	 * but may not be able to model all features of the data. 
	 * 
	 */
	public SelectedTag getCovarianceType() {
		return new SelectedTag(m_CovarianceType, TAGS_COVARIANCE_TYPE);
	}

	/**
	 * set the type of covariance matrices for gaussian HMMs.
	 * A FULL matrix allows arbitrary dependencies between variables, 
	 * a DIAGONAL matrix assumes that all variables are independent. 
	 * SPHERICAL models assume that variables are independent and all have 
	 * the same variance. SPHERICAL covariances imply a more restricted 
	 * set of models than DIAGONAL covariances which are more restricted 
	 * than FULL matrices. Restricted models require less data to learn
	 * but may not be able to model all features of the data. 
	 * 
	 * @param covarianceType the covariance type (possibilities are COVARIANCE_FULL, COVARIANCE_DIAGONAL, COVARIANCE_SPHERICAL)
	 */
	public void setCovarianceType(SelectedTag covarianceType) {
		if (covarianceType.getTags() == TAGS_COVARIANCE_TYPE) {
			m_CovarianceType = covarianceType.getSelectedTag().getID();
		}
	}

	protected boolean m_LeftRight = false;
	
	/**
	 * gets whether the HMM has a Left-Right state structure
	 * In a <em>left-right</em> HMM the sequences of states are always
	 * passed through in a fixed order (possibly missing 
	 * some states). The alternative is an <em>ergodic model</em> (default) in which
	 * arbitrary transitions can be made. 
	 */
	public boolean isLeftRight() {
		return m_LeftRight;
	}

	/**
	 * sets whether the HMM has a Left-Right state structure
	 * In a <em>left-right</em> HMM the sequences of states are always
	 * passed through in a fixed order (possibly missing 
	 * some states). The alternative is an <em>ergodic model</em> (default) in which
	 * arbitrary transitions can be made. 
	 * 
	 * @param leftRight if true the HMM will be left-right, if false it will be ergodic
	 */
	public void setLeftRight(boolean leftRight) {
		this.m_LeftRight = leftRight;
	}

	/*
	 * gets the dimensionality of the HMM outputs (observations)
	 */
	public int getOutputDimension() {
		return m_OutputDimension;
	}

	protected void setOutputDimension(int OutputDimension) {
		m_OutputDimension = OutputDimension;
	}

	/*
	 * gets whether the outputs are numeric
	 * 
	 * @return true if numeric, false if nominal
	 */
	public boolean isNumeric() {
		return m_Numeric;
	}

	protected void setNumeric(boolean Numeric) {
		m_Numeric = Numeric;
		if (m_Numeric)
			setIterationCutoff(0.0001);
		else
			setIterationCutoff(0.01);
	}

	/*
	 * gets the cut off value of terminating the EM iteration.
	 * The EM iteration will stop when the relative change in likelihood 
	 * between two subsequent iterations falls below this cutoff
	 * 
	 * @return the cutoff value
	 */
	public double getIterationCutoff() {
		return m_IterationCutoff;
	}

	/*
	 * sets the cut off value of terminating the EM iteration.
	 * The EM iteration will stop when the relative change in likelihood 
	 * between two subsequent iterations falls below this cutoff
	 * 
	 * @param iterationCutoff the cutoff value
	 */
	public void setIterationCutoff(double iterationCutoff) {
		m_IterationCutoff = iterationCutoff;
	}

	protected HMMEstimator estimators[];

	/**
	 * gets the number of classes for the HMM classifier. 
	 * The value is set automatically from the data
	 */
	public int getNumClasses()
	{
		if (estimators == null)
			return 0; 
		else
			return estimators.length;
	}
	
	/**
	 * gets the number of hidden states in the HMM
	 */
	public int getNumStates() {
		return m_NumStates;
	}

	/**
	 * sets the number of hidden states in the HMM.
	 * The number of states is the major parameter of an HMM and must be set 
	 * (it is not learnt from the data).
	 * 
	 * @param numStates the number of states to be used
	 */
	public void setNumStates(int numStates) {
		m_NumStates = numStates;
	}
	
	/**
	 * gets the number distinct output (observation) values for nominal data
	 * This is set automatically from data.
	 */
	public int getNumOutputs()
	{
		return m_NumOutputs;
	}
	
	protected void setNumOutputs(int numOutputs)
	{
		m_NumOutputs = numOutputs;
	}

	/**
	 * sets a step 0 state/output probability
	 * 
	 * @param classId the class for which we are setting the probability
	 * @param state the current state
	 * @param output the output value
	 * @param prob the probability value to set it to
	 */
	public void setProbability0(int classId, double state, DoubleVector output, double prob) {
		estimators[classId].addValue0(state, output, prob);
	}

	/**
	 * sets a state/output probability
	 * 
	 * @param classId the class for which we are setting the probability
	 * @param prevState the previous state
	 * @param state the current state
	 * @param output the output value
	 * @param prob the probability value to set it to
	 */
	public void setProbability(int classId, double prevState, double state, DoubleVector output, double prob) {
		estimators[classId].addValue(prevState, state, output, prob);
	}
	
	protected double likelihoodFromScales(double scales[])
	{
		double lik = 0.0f;
		for (int i = 0; i < scales.length; i++)
			if (Math.abs((scales[i])) > 1.0E-32)
				lik += Math.log(scales[i]);
			else
				lik += Math.log(1.0E-32);
		return lik;
	}
	
	protected double [] forward(HMMEstimator hmm, Instances sequence, double alpha[][]) throws Exception
	{
		double scales [] =  new double [sequence.numInstances()];
		
		// initial time step
		scales[0] = 0.0f;
		DoubleVector output = new DoubleVector(sequence.instance(0).numAttributes());
		for (int i = 0; i < sequence.instance(0).numAttributes(); i++)
			output.set(i, sequence.instance(0).value(i));
		for (int s = 0; s < m_NumStates; s++)
		{
			alpha[0][s] = hmm.getProbability0(s, output);
			scales[0] += alpha[0][s];
		}
		
		// do scaling
		if (Math.abs(scales[0]) > minScale)
		{
			for (int s = 0; s < m_NumStates; s++)
				alpha[0][s] /= scales[0];
		}
		else
		{
			throw new ProbabilityTooSmallException("time step 0 probability " + scales[0]);
		}
		
		// the rest of the sequence
		for (int t = 1; t < sequence.numInstances(); t++)
		{
			output = new DoubleVector(sequence.instance(t).numAttributes());
			for (int i = 0; i < sequence.instance(t).numAttributes(); i++)
				output.set(i, sequence.instance(t).value(i));
			//System.out.println(output);
			scales[t] = 0.0f;
			for (int s = 0; s < m_NumStates; s++)
			{
				alpha[t][s] = 0.0f;
				for (int ps = 0; ps < m_NumStates; ps++)
				{
					alpha[t][s] += alpha[t-1][ps]*hmm.getProbability(ps, s, output);
				}
				scales[t] += alpha[t][s];
			}
			// do scaling
			if (Math.abs(scales[t]) > minScale)
			{
				for (int s = 0; s < m_NumStates; s++)
					alpha[t][s] /= scales[t];
			}
			else
			{
				throw new ProbabilityTooSmallException("time step " + t + " probability " + scales[t]);
			}
		}
		
		return scales;
	}
	
	/**
	 * evaluate the forward algorithm (calculating likelihood) on a particular sequence
	 * 
	 * @param hmm the hmm estimator to use when evaluating
	 * @param sequence the sequence to evaluate
	 * @return the likelihood of the sequence
	 */
	protected double forward(HMMEstimator hmm, Instances sequence) throws Exception
	{
		double alpha[][] = new double[sequence.numInstances()][m_NumStates];
		double scales[] = forward(hmm, sequence, alpha);
		return likelihoodFromScales(scales);
	}
	
	protected double [] forwardBackward(HMMEstimator hmm, Instances sequence, double alpha[][], double beta[][]) throws Exception
	{
		// do the forward pass
		double scales[] =  forward(hmm, sequence, alpha);
		
		// final time step
		for (int s = 0; s < getNumStates(); s++)
		{
			beta[sequence.numInstances()-1][s] = 1.0f;
			if (Double.isInfinite(beta[sequence.numInstances()-1][s]) || Double.isNaN(beta[sequence.numInstances()-1][s]))
				throw new Exception("Beta for the final timestep is NaN");
		}
		
		// backward through the rest of the sequence
		for (int t = sequence.numInstances()-2; t >= 0; t--)
		{
			for (int s = 0; s < getNumStates(); s++)
			{
				beta[t][s] = 0.0f;
				for (int ns = 0; ns < getNumStates(); ns++)
				{
					DoubleVector output = new DoubleVector(sequence.instance(t+1).numAttributes());
					for (int i = 0; i < sequence.instance(t+1).numAttributes(); i++)
						output.set(i, sequence.instance(t+1).value(i));
					double p = hmm.getProbability(s, ns, output);
					beta[t][s] += beta[t+1][ns]*p;
					if (Double.isInfinite(beta[t][s]) || Double.isNaN(beta[t][s]))
						throw new Exception("Unscaled Beta is NaN");
				}
			}
			if (Math.abs(scales[t+1]) > minScale)
			{
				for (int s = 0; s < getNumStates(); s++)
				{
					beta[t][s] /= scales[t+1];
					if (Double.isInfinite(beta[t][s]) || Double.isNaN(beta[t][s]))
						throw new Exception("Scaled Beta is NaN");
				}
			}
			else
			{
				throw new ProbabilityTooSmallException("time step " + (t+1) + " probability " + scales[t+1]);
			}
		}
		
		return scales;
	}
	
	protected double forwardBackward(HMMEstimator hmm, Instances sequence) throws Exception
	{
		double alpha[][] = new double[sequence.numInstances()][m_NumStates];
		double beta[][] = new double[sequence.numInstances()][m_NumStates];
		double scales [] = forwardBackward(hmm, sequence, alpha, beta);
		return likelihoodFromScales(scales);
	}
	
	/**
	 * evaluate the state occupancy probabilities for a particular sequence
	 * 
	 * @param classId the class for which to do the evaluation
	 * @param instance the data instance (sequence) to evaluate 
	 * @return a two dimensional array of probabilities (doubles). Item [i][j] gives the probability of being in state j for the ith item in the sequence.
	 */
	public double [][] probabilitiesForInstance(int classId, weka.core.Instance instance) throws Exception 
	{
		Instances sequence = instance.relationalValue(m_SeqAttr);
		double alpha[][] = new double[sequence.numInstances()][m_NumStates];
		double beta[][] = new double[sequence.numInstances()][m_NumStates];
		double gamma[][] = new double[sequence.numInstances()][m_NumStates];
		double scales [] = forwardBackward(estimators[classId], sequence, alpha, beta);
		double scale = Math.exp(likelihoodFromScales(scales));
		for (int i = 0; i < gamma.length; i++)
		{
			//System.out.print("gamma {");
			for (int j = 0; j < gamma[i].length; j++)
			{
				gamma[i][j] = alpha[i][j]*beta[i][j];///scale;
				//System.out.print(gamma[i][j] + " ");
			}
			//System.out.println("}");
			//System.out.print("alpha {");
			//for (int j = 0; j < alpha[i].length; j++)
			//{
			//	System.out.print(alpha[i][j] + " ");
			//}
			//System.out.println("}");
			//System.out.print("beta {");
			//for (int j = 0; j < alpha[i].length; j++)
			//{
			//	System.out.print(beta[i][j] + " ");
			//}
			//System.out.println("}");
		}
		return gamma;
	}
	
	/**
	 * get the probabilities from a particular sequence
	 * 
	 * @param instance the data instance (sequence)
	 * @return an array of probabilities (doubles). The nth value is the probability that the sequence is of the nth class.
	 */
	public double[] distributionForInstance(weka.core.Instance instance) throws Exception {
		if (estimators == null)
		{
			double result[] = {0.5,0.5};
			return result;
		}
		
		double [] result = new double[estimators.length];
		double sum = 0.0;
		 
		// means we haven't trained so the results would be garbage
		if (m_SeqAttr < 0)
		{
			for (int j = 0; j < estimators.length; j++)
			{
				result[j] = 1.0;
				sum += result[j];
			};
		}
		else
		{
			Instances seq = instance.relationalValue(m_SeqAttr);
			for (int j = 0; j < estimators.length; j++)
			{
				try
				{
					result[j] = Math.exp(forward(estimators[j], seq));
				}
				catch (ProbabilityTooSmallException e)
				{
					result[j] = 0;
				}
				sum += result[j];
			};
		}
		 
		if (Math.abs(sum) > 0.0000001)
		{
			for (int i = 0; i < estimators.length; i++)
				result[i] /= sum;
		}
		return result;
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return super.clone();
	}

	/**
	   * Returns a string describing this classifier
	   * @return a description of the classifier suitable for
	   * displaying in the explorer/experimenter gui
	   */
	public String globalInfo() {
		return "Class for a Hidden Markov Model classifier.";
	}
	
	@Override
	public String[] getOptions() {
    Vector<String> options = new Vector<String>();

		options.add("-S");
    options.add("" + getNumStates());
	    
		options.add("-I");
    options.add("" + getIterationCutoff());
	    
		options.add("-C");
		switch (m_CovarianceType)
		{
			case MultivariateNormalEstimator.COVARIANCE_FULL:
				options.add("FULL");
				break;
			case MultivariateNormalEstimator.COVARIANCE_DIAGONAL:
				options.add("DIAGONAL");
				break;
			case MultivariateNormalEstimator.COVARIANCE_SPHERICAL:
				options.add("SPHERICAL");
				break;
		}
		if (isTied()) options.add("-D");
		if (isLeftRight()) options.add("-L");
		if (isRandomStateInitializers()) options.add("-R");

    Collections.addAll(options, super.getOptions());
	
		return options.toArray(new String[0]);
	}

	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(6);
		
		newVector.addElement(
			new Option("\tStates: number of HMM states to use\n",
				"S", 1,"-S"));

		newVector.addElement(
			new Option("\tIteration Cutoff: the proportional minimum change of likelihood\n"
				+"\tat which to stop the EM iteractions ",
				"I", 1,"-I"));

		newVector.addElement(
			new Option("\tCovariance Type: whether the covariances of gaussian\n"
				+"\toutputs should be full matrices or limited to diagonal\n"
				+"\tor spherical matrices ",
				"C", 1,"-C"));

		newVector.addElement(
			new Option("\tTied Covariance: whether the covariances of gaussian\n"
				+"\toutputs are tied to be the same across all outputs ",
				"D", 0,"-D"));

		newVector.addElement(
			new Option("\tLeft Right: whether the state transitions are constrained\n"
				+"\tto go only to the next state in numerical order ",
				"L", 0,"-L"));

		newVector.addElement(
			new Option("\tRandom Initialisation: whether the state transition probabilities are intialized randomly\n"
				+"\t(if this is false they are initialised by performing a k-means clustering on the data) ",
				"R", 0,"-R"));
	    
		return newVector.elements();
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		String cutoffString = Utils.getOption('I', options);
		if (cutoffString.length() != 0) 
			setIterationCutoff(Double.parseDouble(cutoffString));
	    
		String statesString = Utils.getOption('S', options);
		if (statesString.length() != 0) 
			setNumStates(Integer.parseInt(statesString));
	    
		String covTypeString = Utils.getOption('C', options);
		if (covTypeString.length() != 0) 
		{
			if (covTypeString.equals("FULL"))
				setCovarianceType(new SelectedTag(MultivariateNormalEstimator.COVARIANCE_FULL, TAGS_COVARIANCE_TYPE));
			if (covTypeString.equals("DIAGONAL"))
				setCovarianceType(new SelectedTag(MultivariateNormalEstimator.COVARIANCE_DIAGONAL, TAGS_COVARIANCE_TYPE));
			if (covTypeString.equals("SPHERICAL"))
				setCovarianceType(new SelectedTag(MultivariateNormalEstimator.COVARIANCE_SPHERICAL, TAGS_COVARIANCE_TYPE));
		}
	      
		if (Utils.getFlag('D', options))
			setTied(true);
	    
		if (Utils.getFlag('L', options))
			setLeftRight(true);
	    
		if (Utils.getFlag('R', options))
			setRandomStateInitializers(true);

    super.setOptions(options);
	    
		Utils.checkForRemainingOptions(options);
	}

	/**
	   * Returns default capabilities of the classifier.
	   *
	   * @return the capabilities of this classifier
	   */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.RELATIONAL_ATTRIBUTES);
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.ONLY_MULTIINSTANCE);

		// class
		result.enable(Capability.NOMINAL_CLASS);

		return result;
	}

	@Override
	public Capabilities getMultiInstanceCapabilities() {
		Capabilities result = new Capabilities(this);
		result.disableAll();
		
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		
		result.enable(Capability.NO_CLASS);
		
		return result;
	}
	  
	protected double EMStep(Instances data) throws Exception
	{
		double lik = 0.0f;
		boolean hasUpdated = false;
		
		// set up new estimators that will store the new
		// distributions for this step
		HMMEstimator newEstimators[] = new HMMEstimator[data.numClasses()];
		for (int i = 0; i < data.numClasses(); i++)
		{
			if (isNumeric())
			{
				MultivariateNormalHMMEstimator est =new MultivariateNormalHMMEstimator(getNumStates(), false);
				est.copyOutputParameters((MultivariateNormalHMMEstimator)estimators[i]);
				newEstimators[i]= est;
			}
			else
			{
				newEstimators[i] = new DiscreteHMMEstimator(getNumStates(), getNumOutputs(), false);
			}
		}
		
		int numS0 = 0;
		int numS1 = 0;
		for (int i = 0; i < data.numInstances(); i++)
		{
			Instance inst = data.instance(i);
			
			// skip if the relevant values are missing
			if (inst.isMissing(m_SeqAttr) || inst.classIsMissing())
				continue;
			
			// e step
			Instances sequence = inst.relationalValue(m_SeqAttr);
			double alpha[][] = new double[sequence.numInstances()][m_NumStates];
			double beta[][] = new double[sequence.numInstances()][m_NumStates];
			
			int classNum = (int) inst.value(data.classIndex());
			
			//System.out.println("****** class " + classNum + " *******");
			HMMEstimator hmm = estimators[classNum];
			
			double scales [];
			try
			{
				scales = forwardBackward(hmm, sequence, alpha, beta);
			}
			catch (ProbabilityTooSmallException e)
			{
				continue;
			}
			double PX = this.likelihoodFromScales(scales);
			lik += PX;
			PX = Math.exp(PX);
			
			// m step
			double sumGamma = 0.0;
			DoubleVector output = new DoubleVector(sequence.instance(0).numAttributes());
			for (int a = 0; a < sequence.instance(0).numAttributes(); a++)
				output.set(a, sequence.instance(0).value(a));

			double gamma[][] = new double[getNumStates()][getNumStates()];
			for (int s = 0; s < getNumStates(); s++)
			{
				gamma[0][s] = alpha[0][s]*beta[0][s];
				sumGamma += gamma[0][s];
				
			}
			for (int s = 0; s < getNumStates(); s++)
			{
				if (sumGamma > minScale)
					newEstimators[classNum].addValue0(s, output, gamma[0][s]/sumGamma);
				
				if (Double.isInfinite(gamma[0][s]) || Double.isNaN(gamma[0][s]))
					throw new Exception("Output of the forward backward algorithm gives a NaN");
			}
		
			for (int t = 1; t < sequence.numInstances(); t++)
			{
				sumGamma = 0.0;
				output = new DoubleVector(sequence.instance(t).numAttributes());
				for (int a = 0; a < sequence.instance(t).numAttributes(); a++)
					output.set(a, sequence.instance(t).value(a));
				for (int s = 0; s < getNumStates(); s++)
					for (int ps = 0; ps < getNumStates(); ps++)
					{
						gamma[ps][s] = alpha[t-1][ps]*hmm.getProbability(ps, s, output)*beta[t][s]*scales[t];
						sumGamma += gamma[ps][s];
					}
				for (int s = 0; s < getNumStates(); s++)
					for (int ps = 0; ps < getNumStates(); ps++)
					{
						if (sumGamma > minScale)
						{
							if (classNum == 0 && s == 1 && gamma[ps][s]/sumGamma > 0.01)
							{
								//System.out.println(classNum + " " + s + " " + output + " " + gamma[ps][s]/sumGamma);
								numS1 += 1;
							}
							if (classNum == 0 && s == 0 && gamma[ps][s]/sumGamma > 0.01)
							{
								numS0 += 1;
							}
							newEstimators[classNum].addValue(ps, s, output, gamma[ps][s]/sumGamma);
						}
						
						// check for undefined numerical values
						if (Double.isInfinite(gamma[ps][s]) || Double.isNaN(gamma[ps][s]))
							throw new Exception("Output of the forward backward algorithm gives a NaN");
						
					}
			}
			hasUpdated = true;
		}
		System.out.println("S0 " + numS0 + " S1 " + numS1);
		System.out.println(newEstimators[0]);
		// update the estimators
		if (hasUpdated)
		{
			estimators = newEstimators;
			for (int i = 0; i < estimators.length; i++)
				estimators[i].calculateParameters();
		}
		else
			throw new Exception("Failed to update on EM step");
		return lik/data.numInstances();
	}
	
	/*
	 * Initialise the hmm estimators prior to learning
	 * 
	 * @param numClasses the number of classes (i.e. the number of estimators)
	 * @param data the dataset that will be used for initialization
	 */
	public void initEstimators(int numClasses, Instances data) throws Exception
	{
		if (isNumeric())
			initEstimatorsMultivariateNormal(numClasses, null, null, null, null, data);
		else	
			initEstimatorsUnivariateDiscrete(numClasses, null, null, null);
	}
	
	protected double[][] initState0ProbsUniform(int numClasses)
	{
		double [][] state0Probs = new double[numClasses][getNumStates()];
		for (int i = 0; i < numClasses; i++)
			for (int j = 0; j < getNumStates(); j++)
			{
				state0Probs[i][j] = 1;
			}
		return state0Probs;
	}
	
	protected double[][] initState0ProbsRandom(int numClasses, Random rand)
	{
		double [][] state0Probs = new double[numClasses][getNumStates()];
		for (int i = 0; i < numClasses; i++)
			for (int j = 0; j < getNumStates(); j++)
			{
				state0Probs[i][j] = rand.nextInt(100);
			}
		return state0Probs;
	}
	
	protected double[][] initState0ProbsLeftRight(int numClasses)
	{
		double [][] state0Probs = new double[numClasses][getNumStates()];
		for (int i = 0; i < numClasses; i++)
		{
			state0Probs[i][0] = 1;
			for (int j = 1; j < getNumStates(); j++)
			{
				state0Probs[i][j] = 0;
			}
		}
		return state0Probs;
	}
	
	protected double[][][] initStateProbsUniform(int numClasses)
	{
		double [][][] stateProbs = new double[numClasses][getNumStates()][getNumStates()];
		for (int i = 0; i < numClasses; i++)
			for (int j = 0; j < getNumStates(); j++)
				for (int k = 0; k < getNumStates(); k++)
				{
					if (j == k)
						stateProbs[i][j][k] = 10;
					else
						stateProbs[i][j][k] = 1;
				}
		return stateProbs;
	}
	
	protected double[][][] initStateProbsRandom(int numClasses, Random rand)
	{
		double [][][] stateProbs = new double[numClasses][getNumStates()][getNumStates()];
		for (int i = 0; i < numClasses; i++)
			for (int j = 0; j < getNumStates(); j++)
				for (int k = 0; k < getNumStates(); k++)
				{
					stateProbs[i][j][k] = rand.nextInt(100);
				}
		return stateProbs;
	}
	
	protected double[][][] initStateProbsLeftRight(int numClasses)
	{
		double [][][] stateProbs = new double[numClasses][getNumStates()][getNumStates()];
		for (int i = 0; i < numClasses; i++)
			for (int j = 0; j < getNumStates(); j++)
			{
				for (int k = 0; k < getNumStates(); k++)
				{
					stateProbs[i][j][k] = 0;
				}
			}
		for (int i = 0; i < numClasses; i++)
		{
			for (int j = 0; j < getNumStates()-1; j++)
			{
				stateProbs[i][j][j]   = 90;
				stateProbs[i][j][j+1] = 10;
			}
			stateProbs[i][getNumStates()-1][getNumStates()-1]   = 100;	
		}
		return stateProbs;
	}
	
	protected double[][][] initDiscreteOutputProbsRandom(int numClasses, Random rand)
	{
		double [][][] outputProbs = new double[numClasses][getNumStates()][getNumOutputs()];
		for (int i = 0; i < numClasses; i++)
			for (int j = 0; j < getNumStates(); j++)
				for (int k = 0; k < getNumOutputs(); k++)
				{
					outputProbs[i][j][k] = rand.nextInt(100);
				}
		return outputProbs;
	}
	
	protected void initGaussianOutputProbsRandom(int numClasses, DoubleVector outputMeans[][], Matrix outputVars[][])
	{
		if (outputMeans == null)
		{
			outputMeans = new DoubleVector[numClasses][getNumStates()];
			for (int i = 0; i < numClasses; i++)
				for (int j = 0; j < getNumStates(); j++)
					outputMeans[i][j] = DoubleVector.random(getOutputDimension());			
		}
		
		if (outputVars == null)
		{
			outputVars = new Matrix[numClasses][getNumStates()];
			for (int i = 0; i < numClasses; i++)
				for (int j = 0; j < getNumStates(); j++)
				{
					outputVars[i][j] = Matrix.identity(getOutputDimension(), getOutputDimension());	
					outputVars[i][j].timesEquals(10.0);
				}
		}
		
		for (int i = 0; i < numClasses; i++)
		{
			for (int j = 0; j < getNumStates(); j++)
			{
				MultivariateNormalHMMEstimator est = (MultivariateNormalHMMEstimator)(estimators[i]);
				est.setOutputMean(j,outputMeans[i][j]);
				est.setOutputVariance(j,outputVars[i][j]);
			}
		}
	}
	
	protected void initGaussianOutputProbsAllData(int numClasses, Instances data, DoubleVector outputMeans[][], Matrix outputVars[][]) throws Exception
	{
		MultivariateNormalEstimator [] ests = new MultivariateNormalEstimator[numClasses];
		for (int i = 0; i < numClasses; i++)
			ests[i] = new MultivariateNormalEstimator();
		
		m_SeqAttr = -1;
		m_NumOutputs = 0;
		for (int i = 0; i < data.numAttributes(); i++)
		{
			Attribute attr = data.attribute(i);
			if (attr.isRelationValued())
			{
				if (attr.relation().attribute(0).isNominal())
				{
					m_SeqAttr = attr.index();
					assert(m_SeqAttr == i);
					m_NumOutputs = attr.relation().attribute(0).numValues();
				}
				if (attr.relation().attribute(0).isNumeric())
				{
					m_SeqAttr = attr.index();
					assert(m_SeqAttr == i);
					this.setNumeric(true);
				}
				break;
			}
		}
		
		for (int i = 0; i < data.numInstances(); i++)
		{
			Instance inst = data.instance(i);
			
			// skip if the relevant values are missing
			if (inst.isMissing(m_SeqAttr) || inst.classIsMissing())
				continue;
			
			// e step
			Instances sequence = inst.relationalValue(m_SeqAttr);
			
			int classNum = (int) inst.value(data.classIndex());
			
			for (int j = 0; j < sequence.numInstances(); j++)
			{
				DoubleVector output = new DoubleVector(sequence.instance(j).numAttributes());
				for (int a = 0; a < sequence.instance(j).numAttributes(); a++)
					output.set(a, sequence.instance(j).value(a));
			
				ests[classNum].addValue(output, 1.0);
			}
		}
		for (int i = 0; i < numClasses; i++)
			ests[i].calculateParameters();
		
		for (int i = 0; i < numClasses; i++)
		{
			for (int j = 0; j < getNumStates(); j++)
			{
				MultivariateNormalHMMEstimator est = (MultivariateNormalHMMEstimator)(estimators[i]);
				if (outputMeans == null)
					est.setOutputMean(j,ests[i].getMean());
				else
					est.setOutputMean(j, outputMeans[i][j]);
				if (outputVars == null)
					est.setOutputVariance(j,ests[i].getVariance());
				else
					est.setOutputVariance(j, outputVars[i][j]);
			}
		}
	}
	
	protected void initGaussianOutputProbsCluster(int numClasses, Instances data, DoubleVector outputMeans[][], Matrix outputVars[][]) throws Exception
	{
		m_SeqAttr = -1;
		m_NumOutputs = 0;
		for (int i = 0; i < data.numAttributes(); i++)
		{
			Attribute attr = data.attribute(i);
			if (attr.isRelationValued())
			{
				if (attr.relation().attribute(0).isNominal())
				{
					m_SeqAttr = attr.index();
					assert(m_SeqAttr == i);
				}
				if (attr.relation().attribute(0).isNumeric())
				{
					m_SeqAttr = attr.index();
					assert(m_SeqAttr == i);
				}
				break;
			}
		}
			
		Instances [] flatdata = new Instances[numClasses];
		for (int i = 0; i < data.numInstances(); i++)
		{
			Instance inst = data.instance(i);
			
			// skip if the relevant values are missing
			if (inst.isMissing(m_SeqAttr) || inst.classIsMissing())
				continue;
			
			// e step
			Instances sequence = inst.relationalValue(m_SeqAttr);
			
			int classNum = (int) inst.value(data.classIndex());
			if (flatdata[classNum] == null)
			{
				flatdata[classNum] = new Instances(sequence, sequence.numInstances());
			}
			
			for (int j = 0; j < sequence.numInstances(); j++)
			{
				flatdata[classNum].add(sequence.instance(j));
			}
		}
		
		SimpleKMeans [] kmeans = new SimpleKMeans[numClasses];
		for (int i = 0; i < numClasses; i++)
		{
			kmeans[i] = new SimpleKMeans();
			kmeans[i].setNumClusters(getNumStates());
			kmeans[i].setDisplayStdDevs(true);
			kmeans[i].buildClusterer(flatdata[i]);
			//System.out.print("Kmeans cluster " + i + " sizes ");
			//double [] clusterSizes = kmeans[i].getClusterSizes();
			//for (int j = 0; j < clusterSizes.length; j++)
			//	System.out.print(clusterSizes[j] + " ");
			//System.out.println("");
		}
		
		for (int i = 0; i < numClasses; i++)
		{
			Instances clusterCentroids = kmeans[i].getClusterCentroids();
			Instances clusterStdDevs = kmeans[i].getClusterStandardDevs();
			for (int j = 0; j < getNumStates(); j++)
			{
				MultivariateNormalHMMEstimator est = (MultivariateNormalHMMEstimator)(estimators[i]);
				if (outputMeans == null)
				{
					DoubleVector mean = new DoubleVector(clusterCentroids.instance(j).numAttributes());
					for (int a = 0; a < clusterCentroids.instance(j).numAttributes(); a++)
						mean.set(a, clusterCentroids.instance(j).value(a));
					
					System.out.println("Mean " + j + " " + mean);
					est.setOutputMean(j, mean);
				}
				else
				{
					est.setOutputMean(j, outputMeans[i][j]);
				}
				if (outputVars == null)
				{
					int n = clusterStdDevs.instance(j).numAttributes();
					Matrix sigma = new Matrix(n, n, 0.0);
					for (int a = 0; a < n; a++)
					{
						double s = clusterStdDevs.instance(j).value(a);
						sigma.set(a, a, s*s);
					}
					est.setOutputVariance(j, sigma);
				}
				else
				{
					est.setOutputVariance(j, outputVars[i][j]);
				}
			}
		}
	}
	
	public void initEstimatorsUnivariateDiscrete(int numClasses, double state0Probs[][], double stateProbs[][][], double outputProbs[][][]) throws Exception
	{
		estimators = new HMMEstimator[numClasses];
		
		// random initialization
		Random rand = new Random(getSeed());
		if (state0Probs == null)
		{
			if (isLeftRight())
				state0Probs = initState0ProbsLeftRight(numClasses);
			else if (isRandomStateInitializers())
				state0Probs = initState0ProbsRandom(numClasses, rand);
			else
				state0Probs = initState0ProbsUniform(numClasses);
		}

		if (stateProbs == null)
		{
			if (isLeftRight())
				stateProbs = initStateProbsLeftRight(numClasses);
			else if (isRandomStateInitializers())
				stateProbs = initStateProbsRandom(numClasses, rand);
			else
				stateProbs = initStateProbsUniform(numClasses);
		}

		if (outputProbs == null)
		{
			outputProbs = initDiscreteOutputProbsRandom(numClasses, rand);
		}
		
		for (int i = 0; i < numClasses; i++)
		{
			estimators[i] = new DiscreteHMMEstimator(getNumStates(), getNumOutputs(), false);
			for (int s = 0; s < getNumStates(); s++)
				for (int o = 0; o < getNumOutputs(); o++)
				{
					estimators[i].addValue0(s, o, 100.0*state0Probs[i][s]*outputProbs[i][s][o]);
					for (int ps = 0; ps < getNumStates(); ps++)
						estimators[i].addValue(ps, s, o, 100.0*stateProbs[i][ps][s]*outputProbs[i][s][o]);
				}
		}
	}
	
	public void initEstimatorsMultivariateNormal(int numClasses, double state0Probs[][], double stateProbs[][][], DoubleVector outputMeans[][], Matrix outputVars[][], Instances data) throws Exception
	{
		estimators = new HMMEstimator[numClasses];
		
		// random initialization
		Random rand = new Random(getSeed());
		if (state0Probs == null)
		{
			if (isLeftRight())
				state0Probs = initState0ProbsLeftRight(numClasses);
			else if (isRandomStateInitializers())
				state0Probs = initState0ProbsRandom(numClasses, rand);
			else
				state0Probs = initState0ProbsUniform(numClasses);
		}

		if (stateProbs == null)
		{
			if (isLeftRight())
				stateProbs = initStateProbsLeftRight(numClasses);
			else if (isRandomStateInitializers())
				stateProbs = initStateProbsRandom(numClasses, rand);
			else
				stateProbs = initStateProbsUniform(numClasses);
		}
		
		for (int i = 0; i < numClasses; i++)
		{
			MultivariateNormalHMMEstimator est = new MultivariateNormalHMMEstimator(getNumStates(), false);
			estimators[i] = est;
			est.setCovarianceType(m_CovarianceType);
			est.setTied(isTied());
			est.setState0Probabilities(state0Probs[i]);
			est.setStateProbabilities(stateProbs[i]);
		}
		
		if (data == null)
		{
			initGaussianOutputProbsRandom(numClasses, outputMeans, outputVars);
		}
		else
		{
			if (isLeftRight())
				initGaussianOutputProbsAllData(numClasses, data, outputMeans, outputVars);
			else
				initGaussianOutputProbsCluster(numClasses, data, outputMeans, outputVars);
		}	
		
	}
	
	/*
	 * train an HMM classifier from data
	 * 
	 * @param data the data set to use in training
	 */
	public void buildClassifier(weka.core.Instances data) throws Exception {
		
		System.out.println("starting build classifier");
		// check that we have class data and that it is in the right form
		if (data.classIndex() < 0)
		{
			System.err.println("could not find class index");
			return;
		}
		if (!data.classAttribute().isNominal())
		{
			System.err.println("class attribute is not nominal");
			return;
		}
		System.out.println(data);

		// find the sequence attribute and then use it to 
		// find the number of outputs
		m_SeqAttr = -1;
		m_NumOutputs = 0;
		for (int i = 0; i < data.numAttributes(); i++)
		{
			Attribute attr = data.attribute(i);
			if (attr.isRelationValued())
			{
				if (attr.relation().attribute(0).isNominal())
				{
					m_SeqAttr = attr.index();
					assert(m_SeqAttr == i);
					m_NumOutputs = attr.relation().attribute(0).numValues();
				}
				if (attr.relation().attribute(0).isNumeric())
				{
					m_SeqAttr = attr.index();
					assert(m_SeqAttr == i);
					this.setNumeric(true);
					m_NumOutputs = -1;
					m_OutputDimension = attr.relation().numAttributes();
					break;
				}
				break;
			}
		}
		
		if (estimators == null)
			initEstimators(data.numClasses(), data);
		
		for (int i = 0; i < estimators.length; i++)
			System.out.println(i + " " + estimators[i]);
		
		if (m_SeqAttr < 0)
		{
			System.err.println("Could not find a relational attribute corresponding to the sequence");
			return;
		}
		
		if (data.numInstances() == 0)
		{
			System.err.println("No instances found");
			return;
		}
			
		double prevlik = -10000000.0;
		for (int step = 0; step < 100; step++)
		{
			double lik = EMStep(data);
			//System.out.println("EM step "+ step + " lik " + lik + " lik change " + Math.abs((lik-prevlik)/lik) + " cutoff " + getIterationCutoff());
			if (Math.abs((lik-prevlik)/lik) < getIterationCutoff())  
				break;
			prevlik = lik;
		}
		for (int i = 0; i < estimators.length; i++)
			System.out.println(i + " " + estimators[i]);
	}
	
	/*
	 * sample sequences from the Hidden Markov Model
	 * 
	 * @param numseqs the number of sequences to sample
	 * @param the length of those sequences
	 * @return an Instances object containing the sequences
	 */
	public Instances sample(int numseqs, int length)
	{
		if (m_rand == null)
			m_rand = new Random(getSeed());
		return sample(numseqs, length, m_rand);
	}
	
	/*
	 * sample sequences from the Hidden Markov Model using a specific random number generator
	 * 
	 * @param numseqs the number of sequences to sample
	 * @param the length of those sequences
	 * @param generator the random number generator to use
	 * @return an Instances object containing the sequences
	 */
	public Instances sample(int numseqs, int length, Random generator)
	{
		ArrayList<Attribute> attrs = new ArrayList<Attribute>();
		
		ArrayList<String> seqIds = new ArrayList<String>();
		for (int i = 0; i < numseqs; i++)
			seqIds.add("seq_"+i);
		attrs.add(new Attribute("seq-id", seqIds));
		
		ArrayList<String> classNames = new ArrayList<String>();
		for (int i = 0; i < estimators.length; i++)
			classNames.add("class_"+i);
		attrs.add(new Attribute("class", classNames));
		
		ArrayList<Attribute> seqAttrs = new ArrayList<Attribute>();
		if (isNumeric())
		{
			for (int i = 0; i < getOutputDimension(); i++)
			{
				seqAttrs.add(new Attribute("output_"+i));
			}
		}
		else
		{
			ArrayList<String> outputs = new ArrayList<String>();
			for (int i = 0; i < getNumOutputs(); i++)
				outputs.add("output_"+i);
			seqAttrs.add(new Attribute("output", outputs));
		}
		Instances seqHeader = new Instances("seq", seqAttrs, 0);
		attrs.add(new Attribute("sequence", seqHeader));
		
		Instances seqs = new Instances("test", attrs, numseqs);
		seqs.setClassIndex(1);
		
		for (int seq=0; seq<numseqs; seq++)
		{
			//System.out.println("sampling sequence "+ seq);
			seqs.add(new DenseInstance(3));
			Instance inst = seqs.lastInstance();
			inst.setValue(0, seqIds.get(seq));
			int classId = m_rand.nextInt(classNames.size());
			inst.setValue(1, classNames.get(classId));
			//System.out.print("class "+classId+":");
			
			HMMEstimator est = estimators[classId];
			
			Instances sequence = new Instances(seqIds.get(seq), seqAttrs, length);
			int state = est.Sample0(sequence, generator);
			for (int i = 1; i < length; i++)
			{
				//System.out.println("sample point "+ i);
				state = est.Sample(sequence, state, generator);
			}
			Attribute seqA = seqs.attribute(2);
			inst.setValue(seqA, seqA.addRelation(sequence));
		}
			
		return seqs;
	}
	
	@Override
	public String toString() {
		StringBuffer text = new StringBuffer();
		text.append("Number of States: "+m_NumStates);
		if (m_NumOutputs > 0)
			text.append("\nNumber of Outputs: "+m_NumOutputs);
		else
			text.append("\nNumber of Output Dimensions: "+m_OutputDimension);
		text.append("\n");
		for (int i = 0; i < estimators.length; i++)
			text.append("\nClass " + i + " " + estimators[i]);
		return text.toString();
	}

	public static void main(String [] argv) {
		runClassifier(new HMM(), argv);
	}
}
