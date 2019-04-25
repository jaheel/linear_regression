#include<iostream>
#include<fstream>
#include "line_regression.h"
#include <cmath>

#define TRAIN_NUM 400//测试数据个数
#define FORECAST_NUM 100//预测数据个数
#define FEATURE_NUM 4//数据特征个数
#define ITERATOR_TIME 1000//迭代次数
#define STUDY_A 0.001//学习率
using namespace std;




double predict(double* w, double* data, int feature_num) {
	double sum = 0;
	for (int i = 0; i < feature_num; i++) {
		sum += w[i] * data[i];
	}
	return sum;
}

// 损失函数
double Theta(double **training_set, int featue_num, int training_num, double* w) {
	double sum = 0;
	for (int i = 0; i < training_num; i++) {
		sum += (training_set[i][featue_num] - predict(w, training_set[i], featue_num))*(training_set[i][featue_num] - predict(w, training_set[i], featue_num));
	}
	return sum / (2 * training_num);
}

// 梯度下降
void gradient_descent(double** training_set, int feature_num, int training_num, double* w, double a, int iterator_time) {
	while (iterator_time--) {
		//迭代前J的值
		double temp = Theta(training_set, feature_num, training_num, w);

		double* del_theta = new double[feature_num];
		for (int i = 0; i < feature_num; i++) {
			del_theta[i] = 0;
			for (int j = 0; j < training_num; j++) {
				del_theta[i] += (predict(w, training_set[j], feature_num) - training_set[j][feature_num])*training_set[j][i];
			}
		}
		//w[i]的更新必须等所有的del_theta测算出来了才可以！不然更新的会影响没更新的
		//上述问题在代码内表示即是下面的for循环不能和上面的合并！
		for (int i = 0; i < feature_num; i++)
		{
			w[i]-= a * del_theta[i] / (double)training_num;
		}
		
		//printf("%.3lf\n", Theta(training_set, feature_num, training_num, w));
		delete[] del_theta;
		//迭代后J的值
		double temp1 = Theta(training_set, feature_num, training_num, w);
		
		//两次迭代J的值变化小于0.001，循环终止
		if (fabs(temp1 - temp) < 0.001)
			break;
	}
	cout << "J_Theta=" << Theta(training_set, feature_num, training_num, w) << endl;
	printf("计算结果：\n");
	for (int i = 0; i < feature_num-1; i++) {
		cout <<"Theta_"<<i<<"="<< w[i] << " ";
	}
	cout << "Theta_"<< feature_num - 1 << "=" << w[feature_num - 1]<<endl;
	return;
}

//测试数据测试
void forecast(double **forecast_set, double* w, int feature_num,int forecast_num) {
	cout << "J_Theta=" << Theta(forecast_set, feature_num, forecast_num, w) << endl;
	for (int j = 0; j < forecast_num; j++) 
	{
		double y = w[0];
		for (int i = 1; i < feature_num - 1; i++) 
		{
			y = y + w[i] * forecast_set[j][i];
		}
		cout << endl;

		//cout << "预测值:" << y << "，实际值：" << forecast_set[j][feature_num] << ",误差:" << fabs(y - forecast_set[j][feature_num]) * 100 / y << endl;

	}
}


void feature_normalize(double **feature_set, int feature_num, int training_num) {
	//特征归一化
	// 对于某个特征 x(i)=(x(i)-average(X))/standard_devistion(X)
	// 1、求出特征X在n个样本中的平均值average（X）
	// 2、求出特征X在n个样本中的标准差 standard_devistion(X)
	// 3、对特征X的n个样本中的每个值x（i），使用上述公式进行归一化
	double *average = new double[feature_num];
	double  *stanrd_divition = new double[feature_num];
	for (int i = 1; i < feature_num; i++) {
		double sum = 0;
		for (int j = 0; j < training_num; j++) {
			sum += feature_set[j][i];
		}
		average[i] = sum / training_num;
	}
	for (int i = 1; i < feature_num; i++) {
		double sum = 0;
		for (int j = 0; j < training_num; j++) {
			sum += (feature_set[j][i] - average[i])*(feature_set[j][i] - average[i]);
		}
		stanrd_divition[i] = sqrt((sum / (training_num - 1)));
	}
	for (int i = 1; i < feature_num; i++)
		for (int j = 0; j < training_num; j++) {
			feature_set[j][i] = (feature_set[j][i] - average[i]) / (double)stanrd_divition[i];
		}
	delete[] stanrd_divition;
	delete[] average;
}









void main() {
	int i, j;//中间变量

	double **training_set = new double*[TRAIN_NUM];//测试数据
	for (i = 0; i < TRAIN_NUM; i++)
	{
		training_set[i] = new double[FEATURE_NUM + 1];
	}

	double *w = new double[FEATURE_NUM];
	//数据初始化
	for (i = 0; i < FEATURE_NUM; i++)
		w[i] = 1.0;
	
	
	ifstream infile;
	infile.open("train_data.txt");
	for (i = 0; i < TRAIN_NUM; i++)
	{
		for (j = 0; j < FEATURE_NUM + 1; j++)
		{
			infile >> training_set[i][j];
		}
	}
	
	infile.close();
	cout << "数据输入成功" << endl;
	
	//梯度下降
	gradient_descent(training_set, FEATURE_NUM, TRAIN_NUM, w, STUDY_A, ITERATOR_TIME);
	
	//预测
	double **forecast_set = new double*[FORECAST_NUM];
	for (i = 0; i < FORECAST_NUM; i++)
	{
		forecast_set[i] = new double[FEATURE_NUM + 1];
	}

	infile.open("test_data.txt");
	for (i = 0; i < TRAIN_NUM; i++)
	{
		for (j = 0; j < FEATURE_NUM + 1; j++)
		{
			infile >> forecast_set[i][j];
		}
	}
	infile.close();
	forecast(forecast_set, w, FEATURE_NUM,FORECAST_NUM);
	feature_normalize(forecast_set, FEATURE_NUM, FORECAST_NUM);

	for (i = 0; i < FORECAST_NUM; i++)
	{
		
		cout <<"第"<<i+1<< "组数据 预测值:" << forecast_set[i][FEATURE_NUM] << endl;
	}
	system("pause");
}
