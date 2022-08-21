#include <iostream>
#include <vector>

#include "CGL/vector2D.h"

#include "mass.h"
#include "rope.h"
#include "spring.h"

namespace CGL {

    Rope::Rope(Vector2D start, Vector2D end, int num_nodes, float node_mass, float k, vector<int> pinned_nodes):
        masses(num_nodes), springs(num_nodes-1)
    {
        // TODO (Part 1): Create a rope starting at `start`, ending at `end`, and containing `num_nodes` nodes.

//        Comment-in this part when you implement the constructor
//        for (auto &i : pinned_nodes) {
//            masses[i]->pinned = true;
//        }
        Vector2D step = (end - start) / (num_nodes-1);
        for (int i=0;i<num_nodes;++i) {
            masses[i] = new Mass(start+i*step, node_mass, false);
        }
        for (int i=0;i<num_nodes-1;++i) {
            springs[i] = new Spring(masses[i], masses[i+1], k);
        }
        for (const auto& i: pinned_nodes) {
            masses[i]->pinned = true;
        }
    }

    void Rope::simulateEuler(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // TODO (Part 2): Use Hooke's law to calculate the force on a node
            auto delta = s->m2->position - s->m1->position;
            auto dir = delta.unit();
            auto fa = s->k * dir * (delta.norm() - s->rest_length);
            s->m1->forces += fa;
            s->m2->forces += -fa;
        }

        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                // TODO (Part 2): Add the force due to gravity, then compute the new velocity and position
                auto f = m->forces + gravity - 0.01 * m->velocity;
                auto a = f / m->mass;
                m->velocity += delta_t * a;
                m->position += delta_t * m->velocity;

                // TODO (Part 2): Add global damping
                // Already impl on the blew
            }

            // Reset all forces on each mass
            m->forces = Vector2D(0, 0);
        }
    }

    void Rope::simulateVerlet(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // TODO (Part 3): Simulate one timestep of the rope using explicit Verlet （solving constraints)
            auto delta = s->m2->position - s->m1->position;
            auto correct_v = 0.5 * (delta.norm()-s->rest_length)*delta.unit();
            if (!s->m1->pinned) {
                s->m1->position += correct_v;
            }
            if (!s->m2->pinned) {
                s->m2->position -= correct_v;
            }
        }

        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                Vector2D temp_position = m->position;
                // TODO (Part 3.1): Set the new position of the rope mass
                auto delta = m->position - m->last_position;
                auto a_delta = gravity * delta_t * delta_t;
                
                // TODO (Part 4): Add global Verlet damping
                m->position += 0.99995 * delta + a_delta;
                m->last_position = temp_position;
            }
        }
    }
}
