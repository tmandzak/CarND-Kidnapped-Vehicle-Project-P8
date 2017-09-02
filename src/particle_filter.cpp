/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  //Number of particles is set as a default value in the constructor

  // These lines creates normal (Gaussian) distributions for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Initialize all particles to first position (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1  
  default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {
    Particle p;

    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  default_random_engine gen;

  // For each particle
  for (int i = 0; i < num_particles; i++) {

    // add measurements   
    if (fabs(yaw_rate) > 0.000001) {
      particles[i].x += velocity / yaw_rate 
          * (sin(particles[i].theta + yaw_rate * delta_t)
              - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate
          * (cos(particles[i].theta)
              - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    } else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }

    // add random Gaussian noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.
  
  // For each observed measurement
  for (int i = 0; i < observations.size(); i++) {

    double min = std::numeric_limits<double>::max();
    int id;

    // find the predicted measurement that is closest to the observed measurement
    for (int j = 0; j < predicted.size(); j++) {
      double d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      if (d < min) {
        min = d;
        id = predicted[j].id;
      }
    }

    // assign the observed measurement to this particular landmark
    observations[i].id = id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  
  
  // For each particle
  for (int i = 0; i < num_particles; i++) {

    double x_p = particles[i].x;
    double y_p = particles[i].y;
    double theta_p = particles[i].theta;

    // get landmarks within sensor range of the particle
    vector < LandmarkObs > predictions;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      int id_m = map_landmarks.landmark_list[j].id_i;
      float x_m = map_landmarks.landmark_list[j].x_f;
      float y_m = map_landmarks.landmark_list[j].y_f;

      if (dist(x_p, y_p, x_m, y_m) <= sensor_range) {
        predictions.push_back(LandmarkObs { id_m, x_m, y_m });
      }
    }

    // transform observations to map coordinates
    vector < LandmarkObs > observations_map;

    for (int j = 0; j < observations.size(); j++) {
      int id_obs = observations[j].id;
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;

      double x_map = x_p + cos(theta_p) * x_obs - sin(theta_p) * y_obs;
      double y_map = y_p + sin(theta_p) * x_obs + cos(theta_p) * y_obs;

      observations_map.push_back(LandmarkObs { id_obs, x_map, y_map });
    }

    dataAssociation(predictions, observations_map);

    particles[i].weight = 1;

    // update the weight of the particle using a mult-variate Gaussian distribution
    for (int j = 0; j < observations_map.size(); j++) {

      int id_ass = observations_map[j].id;
      double x_map = observations_map[j].x;
      double y_map = observations_map[j].y;

      double mu_x, mu_y;
      for (int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == id_ass) {
          mu_x = predictions[k].x;
          mu_y = predictions[k].y;
        }
      }

      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];

      // calculate normalization term
      double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));

      // calculate exponent
      double exponent = (pow(x_map - mu_x, 2) / (2 * pow(sig_x, 2))
          + (pow(y_map - mu_y, 2) / (2 * pow(sig_y, 2))));

      // calculate weight using normalization terms and exponent
      double weight = gauss_norm * exp(-exponent);

      particles[i].weight *= weight;
    }
  }

}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine gen;

  // get weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  std::discrete_distribution<> d(weights.begin(), weights.end());
  // resample
  vector < Particle > particles_res;
  for (int i = 0; i < num_particles; i++)
    particles_res.push_back(particles[d(gen)]);

  particles = particles_res;

}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
