#include <iostream>
#include <fstream>
#include <assert.h>
#include <random>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <time.h>
#include <math.h>
#include <list>
#include <algorithm>
#include "LY_NN.h"

#define velocity 3
#define time_change .2
#define time 5
#define u_max 15
#define u_min -15
#define u_int 0
#define pi 3.1415926535

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
	void update_u();
	bool success();
	bool border();

	};


struct Policy{
	double fitness;
	int time_step;

	void policy_init();
	void mutate();
	void evaluate();
	vector<double> weights;
};

void Ship::ship_init(){
	x_init = 1;
	y_init = 1;
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
	angle_velocity = angle_velocity + (u_value - angle_velocity) * (time_change / time);
}

void Ship::update_x_position(){
	x_previous = x_position; 
	x_position = x_previous + (velocity * cos(angle) * time_change);
}

void Ship::update_y_position(){
	y_previous = y_position;
	y_position = y_previous + (velocity * sin(angle) * time_change);
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

	goal_x = 5;
	goal_a_y = 20;
	goal_b_y = 0;
}

int main(){

	int generation_size = 300;
	int populaton_size = 100;

	Ship nn_ship;
	neural_network jake;
	jake.setup(3,5,1);

	jake.set_in_min_max(0,nn_ship.map.x_size);
	jake.set_in_min_max(0,nn_ship.map.y_size);
	jake.set_in_min_max(0,2*pi);

	jake.set_out_min_max(u_min,u_max);

	while (!nn_ship.success()){
		vector<double> ship_state;
		ship_state.push_back(nn_ship.x_position);
		ship_state.push_back(nn_ship.y_position);
		ship_state.push_back(nn_ship.angle);

		jake.set_vector_input(ship_state);
		jake.execute();
		nn_ship.u_value = jake.get_output(0);
		nn_ship.update_angle_velocity();
		nn_ship.update_angle();
		nn_ship.update_x_position();
		nn_ship.update_y_position();

		if (nn_ship.border()){
			cout << "ship be lost matey";
			break;
		}
		cout << nn_ship.x_position <<"    ";
		cout << nn_ship.y_position << endl;

	}
	cout << "Found goal";
	return 0;
}

// Neural Net Time

