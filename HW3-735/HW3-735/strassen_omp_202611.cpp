// 
// Computes the product of two matrices: C = A * B
// using Strassen's algorithm
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <new>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

#define MAX_MATRIX_SIZE	65536
#define TOL 1.0e-12

#define DEBUG 1

// Global variables
int matrix_size; 
int leaf_matrix_size;

// Define Matrix

class Matrix {
    public:
        int  nrows;		// number of rows
        int  ncols;		// number of columns 
        double **elements;	// Matrix elements

	static Matrix strassens_product(Matrix&, Matrix&);
	static Matrix standard_product(Matrix&, Matrix&);
	static Matrix addition(Matrix&, Matrix&);
	static Matrix subtraction(Matrix&, Matrix&);
	static int compare_matrix(Matrix&, Matrix&); 

	Matrix extract_submatrix(int, int, int, int); 
	void update_submatrix(Matrix&, int, int, int, int); 
	void initialize_matrix(double);
//	void free_memory();
	void print_matrix();

	static void matrix_error(int); 

	Matrix();                       // default - creates an empty matrix
	Matrix(const Matrix&);          // copy  - deep copy
	Matrix(Matrix&&) noexcept;      // move  - steals resources
	Matrix& operator=(const Matrix&);       // copy assignment - deep copy
	Matrix& operator=(Matrix&&) noexcept;   // move assignment
	Matrix(int,int); 
	~Matrix(); 

    private:
	double *array;
	static void strassens_product_task(Matrix&, Matrix&, Matrix&);
};

// Strassen's matrix product 
// - return C = A * B
// - implemented only for square matrices where nrows=ncols
// - parallel version: opens a single parallel region; the recursive
//   work is split into OpenMP tasks (see strassens_product_task)
//
Matrix Matrix::strassens_product(Matrix& A, Matrix& B) {

    int n = A.nrows;
    if (A.nrows != A.ncols) Matrix::matrix_error(500); 
    if (B.nrows != B.ncols) Matrix::matrix_error(501); 
    if (A.ncols != B.nrows) Matrix::matrix_error(502); // Matrix not square

    Matrix C;

    if (n <= leaf_matrix_size) {
	C = Matrix::standard_product(A, B); 
    } else {
	// Fork-join: one thread enters the parallel region and generates
	// tasks; the worker splits the recursion level by level.
	#pragma omp parallel
	#pragma omp single
	{
	    Matrix::strassens_product_task(A, B, C);
	}
    }

    return C;
}

// Recursive worker for the parallel Strassen product.
// - meant to be invoked from within a parallel region (from a task).
// - <n> is the size of A, B and the result C.
// - the 7 independent products M1..M7 and the 4 combinations that
//   build C11..C22 are generated as OpenMP tasks so that all threads
//   of the team share the recursive work.
// - the `final` clause stops spawning further tasks once the sub-block
//   reaches leaf size (avoids task overhead on small matrices).
//
void Matrix::strassens_product_task(Matrix& A, Matrix& B, Matrix& C) {

    int n = A.nrows;

    if (n <= leaf_matrix_size) {
	C = Matrix::standard_product(A, B); 
	return;
    }

    // Extract blocks of A: A11, A12, A21, A22
    Matrix A11 = A.extract_submatrix(0, n/2-1, 0, n/2-1);
    Matrix A12 = A.extract_submatrix(0, n/2-1, n/2, n-1);
    Matrix A21 = A.extract_submatrix(n/2, n-1, 0, n/2-1);
    Matrix A22 = A.extract_submatrix(n/2, n-1, n/2, n-1);

    // Extract blocks of B: B11, B12, B21, B22
    Matrix B11 = B.extract_submatrix(0, n/2-1, 0, n/2-1);
    Matrix B12 = B.extract_submatrix(0, n/2-1, n/2, n-1);
    Matrix B21 = B.extract_submatrix(n/2, n-1, 0, n/2-1);
    Matrix B22 = B.extract_submatrix(n/2, n-1, n/2, n-1);

    // Compute products M1, M2, ..., M7 in parallel
    // Inputs and outputs are shared; each task builds its own local copies.
    Matrix M1, M2, M3, M4, M5, M6, M7;

    #pragma omp task shared(A11,A22,B11,B22,M1) final(n/2 <= leaf_matrix_size)
    {
	Matrix M1a = Matrix::addition(A11,A22); 
	Matrix M1b = Matrix::addition(B11,B22);
	Matrix::strassens_product_task(M1a, M1b, M1);
    }

    #pragma omp task shared(A21,A22,B11,M2) final(n/2 <= leaf_matrix_size)
    {
	Matrix M2a = Matrix::addition(A21,A22);
	Matrix::strassens_product_task(M2a, B11, M2);
    }

    #pragma omp task shared(A11,B12,B22,M3) final(n/2 <= leaf_matrix_size)
    {
	Matrix M3b = Matrix::subtraction(B12,B22);
	Matrix::strassens_product_task(A11, M3b, M3);
    }

    #pragma omp task shared(A22,B21,B11,M4) final(n/2 <= leaf_matrix_size)
    {
	Matrix M4b = Matrix::subtraction(B21,B11);
	Matrix::strassens_product_task(A22, M4b, M4);
    }

    #pragma omp task shared(A11,A12,B22,M5) final(n/2 <= leaf_matrix_size)
    {
	Matrix M5a = Matrix::addition(A11,A12); 
	Matrix::strassens_product_task(M5a, B22, M5);
    }

    #pragma omp task shared(A21,A11,B11,B12,M6) final(n/2 <= leaf_matrix_size)
    {
	Matrix M6a = Matrix::subtraction(A21,A11); 
	Matrix M6b = Matrix::addition(B11,B12);
	Matrix::strassens_product_task(M6a, M6b, M6);
    }

    #pragma omp task shared(A12,A22,B21,B22,M7) final(n/2 <= leaf_matrix_size)
    {
	Matrix M7a = Matrix::subtraction(A12,A22); 
	Matrix M7b = Matrix::addition(B21,B22);
	Matrix::strassens_product_task(M7a, M7b, M7);
    }

    // Wait for all products M1..M7 to complete
    #pragma omp taskwait

    // Compute blocks of C: C11, C12, C21, C22 in parallel
    Matrix C11, C12, C21, C22;

    #pragma omp task shared(M1,M4,M7,M5,C11)
    {
	Matrix C11a = Matrix::addition(M1,M4);
	Matrix C11b = Matrix::subtraction(M7,M5);
	C11 = Matrix::addition(C11a, C11b);
    }

    #pragma omp task shared(M3,M5,C12)
    {
	C12 = Matrix::addition(M3,M5);
    }

    #pragma omp task shared(M2,M4,C21)
    {
	C21 = Matrix::addition(M2,M4);
    }

    #pragma omp task shared(M1,M2,M3,M6,C22)
    {
	Matrix C22a = Matrix::subtraction(M1,M2);
	Matrix C22b = Matrix::addition(M3,M6);
	C22 = Matrix::addition(C22a, C22b);
    }

    // Wait for C11..C22 to complete
    #pragma omp taskwait

    // Create the product matrix C
    C = Matrix(n,n); 
    C.update_submatrix(C11, 0, n/2-1, 0, n/2-1);
    C.update_submatrix(C12, 0, n/2-1, n/2, n-1);
    C.update_submatrix(C21, n/2, n-1, 0, n/2-1);
    C.update_submatrix(C22, n/2, n-1, n/2, n-1);
}
 
// Standard matrix product
// - return C = A * B
Matrix Matrix::standard_product(Matrix& A, Matrix& B) {
    if (A.ncols != B.nrows) matrix_error(5); 
    Matrix C(A.nrows,B.ncols); 
    for (int i = 0; i < C.nrows; i++) {
        for (int j = 0; j < C.ncols; j++) {
	    C.elements[i][j] = 0.0;
	    for (int k = 0; k < A.ncols; k++) 
	        C.elements[i][j] += A.elements[i][k]*B.elements[k][j];
	}
    } 
    return C;
}

// Standard matrix addition
// - return C = A + B
Matrix Matrix::addition(Matrix& A, Matrix& B) {
    if (A.nrows != B.nrows) A.matrix_error(8); 
    if (A.ncols != B.ncols) A.matrix_error(9); 
    Matrix C(A.nrows,A.ncols);  
    for (int i = 0; i < C.nrows; i++) {
        for (int j = 0; j < C.ncols; j++) {
	    C.elements[i][j] = A.elements[i][j]+B.elements[i][j];
	}
    } 
    return C;
}

// Standard matrix subtraction
// - return C = A - B
Matrix Matrix::subtraction(Matrix& A, Matrix& B) {
    if (A.nrows != B.nrows) A.matrix_error(88); 
    if (A.ncols != B.ncols) A.matrix_error(98); 
    Matrix C(A.nrows,A.ncols);  
    for (int i = 0; i < C.nrows; i++) {
        for (int j = 0; j < C.ncols; j++) {
	    C.elements[i][j] = A.elements[i][j]-B.elements[i][j];
	}
    } 
    return C;
}

// Compare if matrix is identical to another by checking if 
// their elements are identical within specified tolerance
int Matrix::compare_matrix(Matrix& A, Matrix& B) {
    int error = 0;
    if (A.nrows != B.nrows) return error; 
    if (A.ncols != B.ncols) return error; 
    for (int i = 0; i < A.nrows; i++) {
        for (int j = 0; j < A.ncols; j++) {
            if (fabs(A.elements[i][j] - B.elements[i][j]) > TOL) error = 1;
	}
    }
    return error;
}

// Extract submatrix of A
// - return S = A[row_first:row_last][col_first:col_last]
Matrix Matrix::extract_submatrix(int row_first, int row_last, 
			  	 int col_first, int col_last) {
    Matrix S(row_last-row_first+1, col_last-col_first+1); 
    for (int i = 0; i < S.nrows; i++) {
        for (int j = 0; j < S.ncols; j++) {
	    S.elements[i][j] = elements[row_first+i][col_first+j];
	}
    } 
    return S;
}

// Update submatrix of A with matrix S
// - Copy S into A[row_first:row_last][col_first:col_last]
void Matrix::update_submatrix(Matrix& S, int row_first, int row_last, 
					int col_first, int col_last) {
    if (S.nrows != (row_last-row_first+1)) S.matrix_error(100);
    if (S.ncols != (col_last-col_first+1)) S.matrix_error(101);
    if (nrows < row_last) S.matrix_error(102); 
    if (ncols < col_last) S.matrix_error(103); 
    for (int i = 0; i < S.nrows; i++) {
        for (int j = 0; j < S.ncols; j++) {
	    elements[row_first+i][col_first+j] = S.elements[i][j];
	}
    } 
}

// Initialize matrix
// - for testing purpose only
void Matrix::initialize_matrix(double factor) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            elements[i][j]= i + factor*j;
	}
    }
}

// Print matrix
void Matrix::print_matrix() {
    printf("\n... Printing matrix ... \n");
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
   	    printf(" %8.4f",elements[i][j]);
	}
	printf("\n");
    }
}

// Generic error
void Matrix::matrix_error(int error_number) {
    printf("Error encountered: %d ... aborting\n", error_number);  
    exit(0);
}

// Default constructor – creates an empty 0x0 matrix
Matrix::Matrix() : nrows(0), ncols(0), elements(nullptr), array(nullptr) {}

// Copy constructor – deep copy (required for safe use with OpenMP tasks)
Matrix::Matrix(const Matrix& src) : nrows(src.nrows), ncols(src.ncols),
                                    elements(nullptr), array(nullptr) {
    if (nrows > 0) {
	elements = new double*[nrows];
	array    = new double[nrows * ncols];
	for (int i = 0; i < nrows; i++)
	    elements[i] = &(array[i * ncols]);
	for (int i = 0; i < nrows * ncols; i++)
	    array[i] = src.array[i];
    }
}

// Move constructor – steal resources from rvalue
Matrix::Matrix(Matrix&& src) noexcept : nrows(src.nrows), ncols(src.ncols),
                                        elements(src.elements), array(src.array) {
    src.nrows    = 0;
    src.ncols    = 0;
    src.elements = nullptr;
    src.array    = nullptr;
}

// Copy assignment – deep copy
Matrix& Matrix::operator=(const Matrix& src) {
    if (this != &src) {
	if (src.nrows > 0 && src.array != nullptr) {
	    if (nrows != src.nrows || ncols != src.ncols) {
		delete[] elements;
		delete[] array;
		nrows    = src.nrows;
		ncols    = src.ncols;
		elements = new double*[nrows];
		array    = new double[nrows * ncols];
		for (int i = 0; i < nrows; i++)
		    elements[i] = &(array[i * ncols]);
	    }
	    for (int i = 0; i < nrows * ncols; i++)
		array[i] = src.array[i];
	} else {
	    delete[] elements;
	    delete[] array;
	    nrows    = 0;
	    ncols    = 0;
	    elements = nullptr;
	    array    = nullptr;
	}
    }
    return *this;
}

// Move assignment – steal resources
Matrix& Matrix::operator=(Matrix&& src) noexcept {
    if (this != &src) {
	delete[] elements;
	delete[] array;
	nrows    = src.nrows;
	ncols    = src.ncols;
	elements = src.elements;
	array    = src.array;
	src.nrows    = 0;
	src.ncols    = 0;
	src.elements = nullptr;
	src.array    = nullptr;
    }
    return *this;
}

// Create new matrix
Matrix::Matrix(int num_rows, int num_cols) {
    nrows = num_rows;
    ncols = num_cols;
    elements = new double *[nrows];
    array = new double[nrows*ncols];
    for (int i = 0; i < nrows; i++) elements[i] = &(array[i*ncols]);
}

// Destroy matrix - free dynamically allocated memory
Matrix::~Matrix(){
	delete [] elements; 
	delete [] array; 
}

// =======================================================================
int main(int argc, char *argv[]) {
    time_t start, end;
    double standard_time, strassens_time;

    // Read input, validate
    if (argc != 3) {
        printf("Need two integers as input \n"); 
        printf("Use: <executable_name> <log_2(matrix_size)> <log_2(leaf_matrix_size)>\n"); 
        exit(0);
    }   
    int k = atoi(argv[argc-2]);
    matrix_size = (1 << k); 
    if (matrix_size > MAX_MATRIX_SIZE) {
        printf("Maximum matrix size allowed: %d.\n", MAX_MATRIX_SIZE);
        exit(0);
    };  
    int q = atoi(argv[argc-1]);
    leaf_matrix_size = (1 << q); 
    if (leaf_matrix_size > matrix_size) {
        printf("Leaf matrix size too large, setting to matrix size ...\n");
        leaf_matrix_size = matrix_size;
    };  

    // Initialize matrices A and B 
    Matrix A(matrix_size,matrix_size); A.initialize_matrix(1.0); 
    Matrix B(matrix_size,matrix_size); B.initialize_matrix(-1.0); 

    // ----------------------------------
    // Strassen's matrix multiplication
    start = omp_get_wtime();
    Matrix C = Matrix::strassens_product(A,B); 
    strassens_time = omp_get_wtime() - start;

    printf("Matrix size = %d, Leaf matrix size = %d, Strassen's (s) = %8.4f s,",
            matrix_size, leaf_matrix_size, strassens_time);

    // ----------------------------------
    // Standard matrix multiplication

    // Compute Cstd = A*B - standard matrix multiplication
    if (k < 11) {
    	start = omp_get_wtime();
	    Matrix Cstd = Matrix::standard_product(A,B); 
    	time(&end); 
    	standard_time = omp_get_wtime() - start;

	    printf(" Standard = %8.4f s,", standard_time);

	    int error = Matrix::compare_matrix(C,Cstd);
	    printf(" Error = %d\n", error);

    	if (error != 0) {
	        printf("Houston, we have a problem!\n");
	    }
	} else {
        printf(" Standard = not computed for large matrices \n");
	}
}

