// Graph cuts segmentation implementation by Chris Whiten
// April, 2012
// Requires: OpenCV (2.3.1)
// Compiled with Visual Studio 2010
// Uses the max-flow code implemented by Vladimir Kolmogorov.
// Based on the paper: Interactive Graph Cuts for Optimal Boundary & Region 
// Segmentation of Objects in N-D images, by Boykov and Jolly (ICCV 2001)

// HOW TO USE THIS SOFTWARE:
//
// To load a specific image to be segmented, edit the variable below, IMAGE_PATH, to the path of the target image.
// Once the image has loaded, use the left mouse-button to click and drag some strokes over the background region.
// In a similar fashion, use the right mouse-button to click and drag some strokes over the foreground region.
// Finally, to run the segmentation, hit enter.
// The segmentation should be displayed quickly, in a second window.

// TUNABLE PARAMETERS:
//
// IMAGE_PATH: the path to the target image to be segmented.
// MOUSE_RADIUS: The radius of the circle around a mouse click (in pixels) with which bg/fg pixels are collected.
// HISTOGRAM_BINS: Number of bins used in the histogram for building probability distributions over bg/fg regions.
// SIGMA: Estimation of the camera noise.  10 is high, but it works well enough.  Variance in pixel intensity between two similar pixels.
// LAMBDA: Dictates the relative weight between region energy and smoothness energy.

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include "graph.h"

using namespace std;

const string IMAGE_PATH = "C:/data/graphcuts/fishes.jpg";

template class Graph<int,int,int>;
template class Graph<short,int,int>;
template class Graph<float,float,float>;
template class Graph<double,double,double>;
typedef int node_id;
typedef Graph<int, int, int> MRF;


bool gathering_bg = false;
bool gathering_fg = false;

MRF *mrf;
cv::Mat *posterior_image;
cv::Mat *input_image;
cv::Mat *input_image_grayscale;
cv::Mat *displayed_image;
cv::Mat *pixels_to_graph;
vector<vector<node_id> > pixel_node_mapping;


const int MOUSE_RADIUS = 5;
const int HISTOGRAM_BINS = 8;
const double BIN_WIDTH = floor(256. / (double)(HISTOGRAM_BINS));
const int SIGMA = 5; // kind of arbitrary right now... This can be estimated as the average camera noise.
const double LAMBDA = 1; // parameter to modify how much the region term affects the total energy.
double K = 0;
const int BACKGROUND_PIXEL = 1;
const int FOREGROUND_PIXEL = 2;

vector<double> bg_hist;
vector<double> fg_hist;

// Return the empircal log-likelihood of a pixel belonging to the given class
// The class distribution is defined over the histogram in arg3
// We keep the histograms normalized, so just extract the correct bin, 
// and take the logarithm.  Simple enough!
double logLikelihood(unsigned row, unsigned col, vector<double> &histogram)
{
	unsigned bin = (int)(floor (input_image_grayscale->at<unsigned char>(row, col) / BIN_WIDTH));
	double probability = histogram[bin];
	return log(probability); // we don't have to worry about 0 logarithms, since we initialize all the bins to have non-zero probabilities.
}

// Initialize the histograms to have non-zero probabilities, 
// since zero probabilities don't really make sense and can lead to math errors.
void initHistograms()
{
	for (unsigned i = 0; i < HISTOGRAM_BINS; ++i)
	{
		bg_hist.push_back(0.001); // this avoids zero probabilities
		fg_hist.push_back(0.001);
	}
}

// Build histograms representing the posterior probabilities
// of a pixel being labeled as foreground or background.
// These probabilities are purely empirical, based on the
// user-generated mouse strokes.
void buildPosteriorHistograms()
{
	unsigned fg_sum = 0;
	unsigned bg_sum = 0;
	initHistograms();

	for (unsigned row = 0; row < posterior_image->rows; ++row)
	{
		for (unsigned col = 0; col < posterior_image->cols; ++col)
		{
			unsigned assignment = posterior_image->at<unsigned>(row, col);
			if (assignment == BACKGROUND_PIXEL)
			{
				unsigned bin = (int)(floor (input_image_grayscale->at<unsigned char>(row, col) / BIN_WIDTH));
				bg_hist.at(bin) += 1;
				bg_sum++;
			}
			else if (assignment == FOREGROUND_PIXEL)
			{
				unsigned bin = (int)(floor (input_image_grayscale->at<unsigned char>(row, col) / BIN_WIDTH));
				fg_hist.at(bin) += 1;
				fg_sum++;
			}
		}
	}

	cout << "normalizing" << endl;
	// normalize
	vector<double>::iterator it;
	for (it = fg_hist.begin(); it != fg_hist.end(); ++it)
	{
		(*it) /= fg_sum;
	}

	for (it = bg_hist.begin(); it != bg_hist.end(); ++it)
	{
		(*it) /= bg_sum;
	}
}



// We are using an 8-neighbourhood system...
// Boundary cost between pixels p and q:
// exp(\frac{- (I_p - I_q)^2}{2\sigma^2}
//
// Compute the costs over a pixel's neighbourhood,
// and add a representative edge between a pixel and
// all of its neighbours in the Markov random field.
double evaluateNeighbourhood(unsigned r, unsigned c)
{
	//cout << "r, c = " << r << ", " << c << endl;
	//cout << input_image_grayscale->cols << endl;
	double cost = 0;
	double B_pq = 0;
	double I_q = 0;
	double I_p = (double)input_image_grayscale->at<unsigned char>(r, c);

	// top-left
	if ((r > 0) && (c > 0))
	{
		I_q = (double)input_image_grayscale->at<unsigned char>(r - 1, c - 1);
		B_pq = exp((-pow(I_p - I_q, 2))/(2 * pow((double)SIGMA, 2)));
		mrf->add_edge(pixels_to_graph->at<int>(r, c), pixels_to_graph->at<int>(r - 1, c - 1), B_pq, B_pq);
		cost += B_pq;
	}

	// top
	if (r > 0)
	{
		I_q = (double)input_image_grayscale->at<unsigned char>(r - 1, c);
		B_pq = exp((-pow(I_p - I_q, 2))/(2 * pow((double)SIGMA, 2)));
		mrf->add_edge(pixels_to_graph->at<int>(r, c), pixels_to_graph->at<int>(r - 1, c), B_pq, B_pq);
		cost += B_pq;
	}

	// top-right
	if ((r > 0) && (c < input_image->cols - 1))
	{
		I_q = (double)input_image_grayscale->at<unsigned char>(r - 1, c + 1);
		B_pq = exp((-pow(I_p - I_q, 2))/(2 * pow((double)SIGMA, 2)));
		mrf->add_edge(pixels_to_graph->at<int>(r, c), pixels_to_graph->at<int>(r - 1, c + 1), B_pq, B_pq);
		cost += B_pq;
	}

	// left
	if (c > 0)
	{
		I_q = (double)input_image_grayscale->at<unsigned char>(r, c - 1);
		B_pq = exp((-pow(I_p - I_q, 2))/(2 * pow((double)SIGMA, 2)));
		mrf->add_edge(pixels_to_graph->at<int>(r, c), pixels_to_graph->at<int>(r, c - 1), B_pq, B_pq);
		cost += B_pq;
	}

	// right
	if (c < input_image_grayscale->cols - 1)
	{
		I_q = (double)input_image_grayscale->at<unsigned char>(r, c + 1);
		B_pq = exp((-pow(I_p - I_q, 2))/(2 * pow((double)SIGMA, 2)));
		mrf->add_edge(pixels_to_graph->at<int>(r, c), pixels_to_graph->at<int>(r, c + 1), B_pq, B_pq);
		cost += B_pq;
	}

	// bottom-left
	if ((r < input_image_grayscale->rows - 1) && (c > 0))
	{
		I_q = (double)input_image_grayscale->at<unsigned char>(r + 1, c - 1);
		B_pq = exp((-pow(I_p - I_q, 2))/(2 * pow((double)SIGMA, 2)));
		mrf->add_edge(pixels_to_graph->at<int>(r, c), pixels_to_graph->at<int>(r + 1, c - 1), B_pq, B_pq);
		cost += B_pq;
	}

	// bottom
	if (r < input_image_grayscale->rows - 1)
	{
		I_q = (double)input_image_grayscale->at<unsigned char>(r + 1, c);
		double inside_exp = (-pow(I_p - I_q, 2))/(2 * pow((double)SIGMA, 2));
		
		B_pq = exp(inside_exp);
		mrf->add_edge(pixels_to_graph->at<int>(r, c), pixels_to_graph->at<int>(r + 1, c), B_pq, B_pq);
		cost += B_pq;
	}

	// bottom-right
	if ((r < input_image_grayscale->rows - 1) && (c < input_image->cols - 1))
	{
		I_q = (double)input_image_grayscale->at<unsigned char>(r + 1, c + 1);
		B_pq = exp((-pow(I_p - I_q, 2))/(2 * pow((double)SIGMA, 2)));
		mrf->add_edge(pixels_to_graph->at<int>(r, c), pixels_to_graph->at<int>(r + 1, c + 1), B_pq, B_pq);
		cost += B_pq;
	}
	return cost;
}

// Set up the pairwise clique potentials for each pairwise neighbour in the MRF.
// At the same time, evaluate which pixel location has the highest cost neighbourhood.
// This high-cost neighbourhood is used to compute the weight K used in singleton cliques.
void setPairwiseCliquePotentials()
{
	double max_neighbourhood_cost = 0;

	for (unsigned row = 0; row < input_image->rows; ++row)
	{
		for (unsigned col = 0; col < input_image->cols; ++col)
		{
			double neighbourhood_cost = evaluateNeighbourhood(row, col);
			max_neighbourhood_cost = max(neighbourhood_cost, max_neighbourhood_cost);
			//cout << "Pairwise cost: " << neighbourhood_cost << endl;
		}
	}
	K = max_neighbourhood_cost;
}

// Set the "region" energy terms in our MRF.
// Generally, just the singleton clique potentials,
// as described in "Interactive Graph Cuts for 
// Optimal Boundary & Region Segmentation of 
// Objects in N-D Images" by Boykov and
// Jolly.
void setSingletonCliquePotentials()
{
	for (unsigned row = 0; row < input_image->rows; ++row)
	{
		for (unsigned col = 0; col < input_image->cols; ++col)
		{
			// seeded pixels
			if (posterior_image->at<unsigned>(row, col) == BACKGROUND_PIXEL)
			{
				mrf->add_tweights(pixels_to_graph->at<int>(row, col), 0, K);
			}
			else if (posterior_image->at<unsigned>(row, col) == FOREGROUND_PIXEL)
			{
				mrf->add_tweights(pixels_to_graph->at<int>(row, col), K, 0);
			}

			// unmarked pixels
			else
			{
				// these energys are the -log likelihoods of being in either class.
				// these probabilities are drawn empirically from the histograms.
				double region_energy_bg = -logLikelihood(row, col, bg_hist) * LAMBDA;
				double region_energy_fg = -logLikelihood(row, col, fg_hist) * LAMBDA;
				mrf->add_tweights(pixels_to_graph->at<int>(row, col), region_energy_bg, region_energy_fg);
			}
		}
	}
}

// Display the computed segmentation.
void visualizeSegmentation()
{
	//(g->what_segment(1) == GraphType::SOURCE)
	cv::Mat visualize(input_image->rows, input_image->cols, CV_32F, cv::Scalar(0));

	for (unsigned row = 0; row < input_image->rows; ++row)
	{
		for (unsigned col = 0; col < input_image->cols; ++col)
		{
			node_id id = pixels_to_graph->at<int>(row, col);
			if (mrf->what_segment(id) == MRF::SOURCE)
			{
				//cout << "Source" << endl;
				visualize.at<float>(row, col) = 0;
			}
			else
			{
				//cout << "Sink" << endl;
				visualize.at<float>(row, col) = 255;
			}
		}
	}

	cv::namedWindow("Segmentation");
	cv::imshow("Segmentation", visualize);
	//cv::waitKey();
}

void runSegmentation()
{
	cout << "Computing posterior probabilities" << endl;
	buildPosteriorHistograms();
	cout << "Computing pairwise clique energies" << endl;
	setPairwiseCliquePotentials();
	cout << "Computing singleton clique energies" << endl;
	setSingletonCliquePotentials();
	int flow = mrf->maxflow();
	cout << "Maximum flow: " << flow << endl;
	visualizeSegmentation();
}


// Update the GUI to visualize user mouse strokes.
void updateDisplay()
{
	cout << "here" << endl;
	int count = 0;
	for (unsigned row = 0; row < input_image->rows; ++row)
	{
		for (unsigned col = 0; col < input_image->cols; ++col)
		{
			// if not background or foreground, just display original image here.
			if (posterior_image->at<unsigned>(row, col) == FOREGROUND_PIXEL)
			{
				displayed_image->at<cv::Vec3b>(row, col)[0] = 0;//input_image->at<cv::Vec3b>(row, col)[0];
				displayed_image->at<cv::Vec3b>(row, col)[1] = 100;//input_image->at<cv::Vec3b>(row, col)[1];
				displayed_image->at<cv::Vec3b>(row, col)[2] = input_image->at<cv::Vec3b>(row, col)[2];
				count++;
			}
			else if (posterior_image->at<unsigned>(row, col) == BACKGROUND_PIXEL)
			{
				displayed_image->at<cv::Vec3b>(row, col)[0] = 100;// input_image->at<cv::Vec3b>(row, col)[0];
				displayed_image->at<cv::Vec3b>(row, col)[1] = input_image->at<cv::Vec3b>(row, col)[1] * 0.7;// + 100;
				displayed_image->at<cv::Vec3b>(row, col)[2] = 100;//input_image->at<cv::Vec3b>(row, col)[2];
			}
			else
			{
				displayed_image->at<cv::Vec3b>(row, col)[0] = input_image->at<cv::Vec3b>(row, col)[0];
				displayed_image->at<cv::Vec3b>(row, col)[1] = input_image->at<cv::Vec3b>(row, col)[1];
				displayed_image->at<cv::Vec3b>(row, col)[2] = input_image->at<cv::Vec3b>(row, col)[2];
			}
		}
	}

	cv::imshow("Display", *displayed_image);
}

// Handle mouse clicks.
void onMouse(int event, int x, int y, int, void *)
{
	switch (event)
	{
		// gather background points with left-mouse button.
		case CV_EVENT_LBUTTONDOWN:
			gathering_bg = true;
			break;
			// update visualization.

		case CV_EVENT_LBUTTONUP:
			gathering_bg = false;
			// update visualization.
			updateDisplay();
			//runSegmentation();
			break;

		// gather foreground points with the right-mouse button.
		case CV_EVENT_RBUTTONDOWN:
			gathering_fg = true;
			break;

		case CV_EVENT_RBUTTONUP:
			gathering_fg = false;
			updateDisplay();
			//runSegmentation();
			break;

		case CV_EVENT_MOUSEMOVE:
			if (gathering_bg || gathering_fg)
			{
				// add points from current x,y location to bg/fg list.
				for (int dx = -MOUSE_RADIUS; dx <= MOUSE_RADIUS; ++dx)
				{
					for (int dy = -MOUSE_RADIUS; dy <= MOUSE_RADIUS; ++dy)
					{
						int observed_x = x + dy;
						int observed_y = y + dx;
						// make sure we're within the borders of the image.
						if ((observed_x >= 0) && (observed_x < posterior_image->cols) && (observed_y >= 0) && (observed_y < posterior_image->rows))
						{
							int pixel_value;
							(gathering_bg) ? (pixel_value = BACKGROUND_PIXEL) : (pixel_value = FOREGROUND_PIXEL);
							posterior_image->at<unsigned>(observed_y, observed_x) = pixel_value;
						}
					}
				}
			}
	}
}

// Initialize all of the requisite structures.
void init()
{
	// load image into needed matrices
	/////////////////////////////////////////////
	cv::Mat tmp = cv::imread(IMAGE_PATH);
	input_image = new cv::Mat(tmp);
	displayed_image = new cv::Mat(tmp);
	input_image_grayscale = new cv::Mat(tmp);
	cv::cvtColor(*input_image_grayscale, *input_image_grayscale, CV_BGR2GRAY);


	// initialize structure that 
	// holds observed fg/bg pixels.
	/////////////////////////////////////////////
	posterior_image = new cv::Mat(input_image->rows, input_image->cols, CV_32S, cv::Scalar(0));
	initHistograms();

	// initialize structure to hold 
	// mapping from pixels to nodes in the graph
	/////////////////////////////////////////////
	int num_nodes = input_image->rows * input_image->cols;
	int num_edges = (4*(input_image->rows)*(input_image->cols)) - (3 * (input_image->cols)) - (3 * (input_image->rows)) + 2;

	mrf = new MRF(num_nodes, num_edges); 
	pixels_to_graph = new cv::Mat(input_image->rows, input_image->cols, CV_32S);
	for (int row = 0; row < pixels_to_graph->rows; ++row)
	{
		for (int col = 0; col < pixels_to_graph->cols; ++col)
		{
			node_id id = mrf->add_node();
			pixels_to_graph->at<int>(row, col) = id;
		}
	}

	cv::namedWindow("Display");
	cv::namedWindow("Segmentation");
	cv::imshow("Display", *input_image);

	//runSegmentation();
}

void main()
{
	init();
	// this mouse callback will handle gathering bg/fg pixels until a key is pressed.
	cv::setMouseCallback("Display", onMouse, 0);

	// once a key is pressed, we run the segmentation.
	cv::waitKey();

	runSegmentation();
	cv::waitKey();
	delete mrf;

}