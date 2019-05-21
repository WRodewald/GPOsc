#pragma once

#include <vector>
#include <Eigen/Dense>

#include "GPVarianceKernel.h"

namespace GP
{
	

	class Model
	{
	public:
		Model(std::vector<std::pair<float, float>> points, const VarianceKernel &varKernel, unsigned int size);
		

	public:

		unsigned int getSize() const;

		const Eigen::VectorXf & getMean()   const;
		const Eigen::MatrixXf & getVar()    const;
		const Eigen::MatrixXf & getVarInv() const;


	private:
		const unsigned int Size;

		Eigen::VectorXf mean;
		Eigen::MatrixXf var;
		Eigen::MatrixXf var_inv;
	};

}