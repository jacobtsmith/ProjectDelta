#include <iostream>
#include <fstream>
#include <assert.h>
#include <random>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <math.h>
#include <list>
#include <algorithm>
#include "LY_NN.h"

#define velocity 3
#define time_change .2
#define Time 5
#define u_max .26
#define u_min -.26
#define u_int 0
#define pi 3.1415926535
#define mut_size 5
#define LYRAND (double)rand()/RAND_MAX

using namespace std;

struct Grid{
	int x_size;
	int y_size;

	double goal_x;
	double goal_a_y;
	double goal_b_y;

	void grid_init();

};

class Ship{
	public:
	double x_init;
	double y_init;
	double angle;
	double angle_velocity;
	double x_previous;
	double x_position;
	double y_previous;
	double y_position;
	double u_value;
	Grid map;

	Ship();
	void update_angle();
	void update_angle_velocity();
	void update_x_position();
	void update_y_position();
	void ship_init();
	double target_distance();
	bool success();
	bool border();

	};


class Policy{
public:
	double fitness;
	int time_step;
	vector<double> weights;

	void policy_init(int);
	void mutate();
	void evaluate(double);
};

class Population{
public:
	int pop_size;
	vector<Policy> members;
	Population(int = 22);
	void EA_init(int);
	void binary_elim();
	double average_calc();
};

double Population::average_calc(){
	double sum = 0;
	double avg = 0;
	for (int i=0; i < pop_size; i++){
		sum = sum + members[i].fitness;
		avg = sum / pop_size;
	}
	return avg;	
}

void Policy::evaluate(double f){
	fitness = f;
}

void Policy::mutate(){
    for(int i = 0; i<weights.size(); i++){
        if(rand()%2==0){
            weights.at(i)+= mut_size * LYRAND - mut_size * LYRAND;
        }
    }
}

void Population::binary_elim(){

    random_shuffle(std::begin(members),std::end(members));
    for (int i=0; i < pop_size; i = i+2){
        double fitness_a = members[i].fitness;
        double fitness_b = members[i+1].fitness;
        if (fitness_a < fitness_b){
            members[i+1] = members [i];
            members[i+1].mutate();
        }
        else {
            members[i] = members [i+1];
            members[i].mutate(); 
        }
    }
    random_shuffle(std::begin(members),std::end(members)); 
}

void Policy::policy_init(int num_of_weights){
    vector<double> b;
    for(int i = 0; i<num_of_weights; i++){
        b.push_back(LYRAND-LYRAND);
    }
    weights = b;
}

Population :: Population(int s){
	pop_size = s;
}
void Population::EA_init(int num_weights){
    
    for(int i=0; i<pop_size; i++){
    	Policy new_policy;
    	new_policy.policy_init(num_weights);
    	members.push_back(new_policy);
    }
       
}


void Ship::ship_init(){
	x_init = rand()%100;
	y_init = rand()%100;
	x_previous = x_init;
	y_previous = y_init;
	x_position = x_previous;
	y_position = y_previous;
	angle = 0;
	angle_velocity = 0;

}

Ship::Ship(){
	ship_init();
	map.grid_init();
}

void Ship::update_angle(){
	angle = angle + (angle_velocity * time_change);
}

void Ship::update_angle_velocity(){
	angle_velocity = angle_velocity + (u_value - angle_velocity) * (time_change / Time);
}

void Ship::update_x_position(){
	x_previous = x_position; 
	x_position = x_previous + (velocity * cos(angle) * time_change);
}

void Ship::update_y_position(){
	y_previous = y_position;
	y_position = y_previous + (velocity * sin(angle) * time_change);
}

double Ship::target_distance(){
	double middle_y = (map.goal_a_y - map.goal_b_y)/2;
	return fabs(middle_y - y_position) + fabs(map.goal_x - x_position);
}

bool Ship::border(){
	if(x_position < 0 || x_position > map.x_size || y_position < 0 || y_position > map.y_size){
		return true;
	}
	return false;
}

bool Ship::success(){
	if ((y_position >= map.goal_b_y && y_position <= map.goal_a_y) || (y_previous >= map.goal_b_y && y_previous <= map.goal_a_y)){
		if ((map.goal_x >= x_previous && map.goal_x <= x_position) || (map.goal_x >= x_position && map.goal_x <= x_previous)){
			return true;
		}
		else 
			return false;
	}
	else 
		return false;
}

void Grid::grid_init(){
	x_size = 100;
	y_size = 100;

	goal_x = 50;
	goal_a_y = 25;
	goal_b_y = 0;
}

int main(){

    srand(time(NULL));

	int generation_size = 500;

	Population nn_pop;
	Ship nn_ship;
	neural_network jake;
	jake.setup(4,5,1);

	jake.set_in_min_max(0,nn_ship.map.x_size);
	jake.set_in_min_max(0,nn_ship.map.y_size);
	jake.set_in_min_max(0,2*pi);
	jake.set_in_min_max(0,nn_ship.map.y_size + nn_ship.map.x_size);

	jake.set_out_min_max(u_min,u_max);
	int weight_count = jake.get_number_of_weights();

	nn_pop.EA_init(weight_count);

	for (int i=0; i < generation_size ; ++i){
		for (int j=0; j < nn_pop.pop_size; ++j){
			jake.set_weights(nn_pop.members[j].weights, true);
			int count = 0;
			double distance;

			while (!nn_ship.success()){
				vector<double> ship_state;
				ship_state.push_back(nn_ship.x_position);
				ship_state.push_back(nn_ship.y_position);
				ship_state.push_back(nn_ship.angle);
				distance = nn_ship.target_distance();
				ship_state.push_back(distance);

				jake.set_vector_input(ship_state);
				jake.execute();
				nn_ship.u_value = jake.get_output(0);
				nn_ship.update_angle_velocity();
				nn_ship.update_angle();
				nn_ship.update_x_position();
				nn_ship.update_y_position();

				count++;

				if (count > 500){
					break;
				}
			}
			nn_ship.x_position = nn_ship.x_init;
			nn_ship.y_position = nn_ship.y_init;
			nn_ship.x_previous = nn_ship.x_init;
			nn_ship.y_previous = nn_ship.y_init;

			nn_pop.members[j].evaluate(distance);
		}
		nn_pop.binary_elim();
		cout << nn_pop.average_calc() << "  ";
		cout << endl;

	}
	return 0;
}


