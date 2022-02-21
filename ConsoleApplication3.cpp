#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include <windows.h>
#include <ctime>
#include <chrono>

#define MAX_SIZE 100

using namespace std;

// Inits matrix
void init_matrix(double**);

// Fills matrix with random values
void fill_matrix(size_t, size_t, double**, int);

// Compares two matrix
bool compare(double**, double**);

// Calculation without vect
void no_vec_mul(int, int, int, double**, double**, double**);

// Calculation with vect
void vec_mul(int, int, int, double**, double**, double**);

// Manual calculation
void AVX(int, int, int, double**, double**, double**);

int main() 
{
	double **matrix_A = new double* [MAX_SIZE * 16], **matrix_B = new double* [MAX_SIZE * 16];
	double **matrix_C1 = new double* [MAX_SIZE * 16], **matrix_C2 = new double* [MAX_SIZE * 16], **matrix_AVX = new double* [MAX_SIZE * 16];

	// Initialization of all matrixes
	init_matrix(matrix_A);
	init_matrix(matrix_B);
	init_matrix(matrix_C1);
	init_matrix(matrix_C2);
	init_matrix(matrix_AVX);

	// Filling matrixes
	fill_matrix(MAX_SIZE * 16, MAX_SIZE * 16, matrix_A, 15);
	fill_matrix(MAX_SIZE * 16, MAX_SIZE * 16, matrix_B, 40);

	no_vec_mul(MAX_SIZE * 16, MAX_SIZE * 16, MAX_SIZE * 16, matrix_A, matrix_B, matrix_C1);
	AVX(MAX_SIZE * 16, MAX_SIZE * 16, MAX_SIZE * 16, matrix_A, matrix_B, matrix_AVX);
	vec_mul(MAX_SIZE * 16, MAX_SIZE * 16, MAX_SIZE * 16, matrix_A, matrix_B, matrix_C2);

	if (compare(matrix_C1, matrix_C2) && compare(matrix_C1, matrix_AVX)) 
		cout << endl << "C1 = C2 = AVX";
	else
		cout << endl << "not equal!";

	return 0;
}

void fill_matrix(size_t size_x, size_t size_y, double** matrix, int rand_key)
{
	srand(rand_key);

	for (int i = 0; i < size_x; i++)
		for (int j = 0; j < size_y; j++)
			matrix[i][j] = rand() % 100;
}

void init_matrix(double **new_matrix)
{
	for (int i = 0; i < MAX_SIZE * 16; i++)
	{
		// Sets matrix 2-st dimension
		new_matrix[i] = new double[MAX_SIZE * 16];
		// Inits matrix 2-nd dimension with 0
		ZeroMemory(new_matrix[i], MAX_SIZE * 16 * sizeof(double));
	}
}

bool compare(double** matrix_1, double** matrix_2)
{
	bool flag = TRUE;

	for (int i = 0; i < MAX_SIZE * 16; i++)
		for (int j = 0; j < MAX_SIZE * 16; j++)
			if (matrix_1[i][j] != matrix_2[i][j])
				flag = FALSE;

	return flag;
}

void no_vec_mul(int x_A, int y_B, int y_A, double** A, double** B, double** C)
{
	auto start = chrono::high_resolution_clock::now();

	for (int i = 0; i < x_A; i++)
	{
		double* ans = C[i];

		for (int k = 0; k < y_A; k++)
		{
			const double* mx_b = B[k];
			double mx_a = A[i][k];

#pragma loop(no_vector)

			for (int j = 0; j < y_B; j++)
				ans[j] += mx_a * mx_b[j];
		}
	}

	auto end = chrono::high_resolution_clock::now();

	chrono::duration<double> result = end - start;

	cout << "Without vec mul: " << result.count() << " seconds" << endl;
}

void vec_mul(int x_A, int y_B, int y_A, double** A, double** B, double** C)
{
	auto start = chrono::high_resolution_clock::now();

	for (int i = 0; i < x_A; i++)
	{
		double* mx_c = C[i];

		for (int k = 0; k < y_A; k++)
		{
			const double* mx_b = B[k];
			double mx_a = A[i][k];

			for (int j = 0; j < y_B; j++)
				mx_c[j] += mx_a * mx_b[j];
		}
	}

	auto end = chrono::high_resolution_clock::now();

	chrono::duration<double> result = end - start;

	cout << "Vec mul: " << result.count() << " seconds" << endl;
}

void AVX(int x_A, int y_B, int y_A, double** A, double** B, double** C)
{
	auto start = chrono::high_resolution_clock::now();

	for (int i = 0; i < x_A; i++)
	{
		double* mx_c = C[i];

		for (int k = 0; k < y_A; k++)
		{
			const double* mx_b = B[k];
			const __m256d mx_a = _mm256_set1_pd(A[i][k]);

			for (int j = 0; j < y_B; j += 4)
			{
				_mm256_storeu_pd(mx_c + j,
					_mm256_add_pd(
						_mm256_loadu_pd(mx_c + j),
						_mm256_mul_pd(mx_a, _mm256_loadu_pd(mx_b + j))));
			}
		}
	}

	auto end = chrono::high_resolution_clock::now();

	chrono::duration<double> result = end - start;

	cout << "AVX mul: " << result.count() << " seconds" << endl;
}