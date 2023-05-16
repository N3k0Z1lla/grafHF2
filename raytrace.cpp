//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"
#include <cmath>
#include <cstdio>
#include <math.h>
#include <vector>

struct Hit {
	float t;
	vec3 position, normal;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius) {
		center = _center;
		radius = _radius;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		return hit;
	}
};

struct Triangle : Intersectable {
	vec3 p1, p2, p3;

	Triangle(vec3 p1, vec3 p2, vec3 p3) : p1(p1), p2(p2), p3(p3) {
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 norm = cross(p2 - p1, p3 - p1);
		float t = dot(p1 - ray.start, norm) / dot(ray.dir, norm);
		vec3 p = ray.start + ray.dir * t;
		if(dot(cross(p2 - p1, p - p1), norm) > 0 && 
		dot(cross(p3-p2,p-p2), norm) > 0 && 
			dot(cross(p1 - p3, p - p3), norm) > 0 ) {
			hit.t = t;
			hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(norm); 
		}
	return hit;
	}
};

struct Cone : Intersectable {
	vec3 pos;
	vec3 norm;
	float h;
	float angle;

	Cone(vec3 _pos, vec3 _norm, float _h, float a) {
		pos = _pos;
		norm = _norm;
		h = _h;
		angle = a;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 A = ray.start - pos;
		float B = cosf(angle)*cosf(angle);
		vec3 d = ray.dir;
		float a = dot(d, norm)*dot(d, norm) - dot(d, d)*B;
		float b = 2 * (dot(A, norm) * dot(d, norm) - dot(A, d) * B);
		float c = dot(A, norm) * dot(A, norm) - dot(A, A) * B;
		float diskriminant = (b*b - 4 * a * c);
		if(diskriminant < 0) {
			//printf("elso = %f\nmasodik = %f\nharadik = %f\n", a, b, c);
		//	printf("disc\n");
			return hit;
		}
		float t1 = (-b + (sqrt(diskriminant))) / (2 * a);
		float t2 = (-b - (sqrt(diskriminant))) / (2 * a);
		float vicces1 = dot((A + d*t1), norm);
		float vicces2 = dot((A + d*t2), norm);
		if(0 <= vicces1 && vicces1 <= h && t1 > 0) {
			hit.t = t1;
			//printf("1es\n");
			hit.position = ray.start + ray.dir * hit.t;
			hit.normal = 
				2 * dot(hit.position - pos, norm) * norm - 2*(hit.position - pos)*B; 
			normalize(hit.normal);
			return hit;
		}
		if(0 <= vicces2 && vicces2 <= h) {
			hit.t = t2;
			//printf("2es\n");
			hit.position = ray.start + ray.dir * hit.t;
			hit.normal = 
				2 * dot(hit.position - pos, norm) * norm - 2*(hit.position - pos)*B; 
			normalize(hit.normal);
			return hit;
		}
	//	printf("sima\n");
		return hit;
	}
};


class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 pos, color;

	Light(vec3 pos, vec3 color) : pos(pos), color(color) {}
};

std::vector<Triangle> helper(std::vector<vec3> vertices, std::vector<std::vector<int>> indecies, vec3 offset, float scale) {
	std::vector<Triangle> trigs;

	for(auto xd : indecies) {
		trigs.emplace_back(scale * vertices[xd[0]] + offset, scale * vertices[xd[1]] + offset, scale * vertices[xd[2]] + offset);
	}
	return trigs;
}



std::vector<Triangle> cubeMaker() {
	std::vector<vec3> vertices = {
		vec3(0.0,  0.0,  0.0),
		vec3(0.0,  0.0,  1.0),
		vec3(0.0,  1.0,  0.0),
		vec3(0.0,  1.0,  1.0),
		vec3(1.0,  0.0,  0.0),
		vec3(1.0,  0.0,  1.0),
		vec3(1.0,  1.0,  0.0),
		vec3(1.0,  1.0,  1.0),	
	};
	
	std::vector<std::vector<int>> indexes = {
		{0,  6,  4},
		{0,  2,  6}, 
		{0,  3,  2}, 
		{0,  1,  3}, 
		{2,  7,  6}, 
		{2,  3,  7}, 
		{0,  4,  5}, 
		{0,  5,  1}, 
		};

	return helper(vertices, indexes, vec3(-0.5, -0.5, -0.5), 1);
}

std::vector<Triangle> tetrahedronMaker() {
	std::vector<vec3> vertices = {
		vec3(1.00, 1.00, 1.00),
		vec3(2.00, 1.00, 1.00),
		vec3(1.00, 2.00, 1.00),
		vec3(1.00, 1.00, 2.00),
	};

	std::vector<std::vector<int>> indexes = {
		{1, 3, 2},
		{1, 4, 3},
		{1, 2, 4},
		{2, 3, 4},
	};

	return helper(vertices, indexes, vec3(-0.5, -0.5, -0.5), 0.2);
}


float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(sqrt(2), 0, sqrt(2)), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0, 0, 0);
		lights.push_back(new Light(vec3(0, 0, 0), vec3(1, 0, 0)));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		//objects.push_back(new Cone(vec3(0, 0, 0), vec3(sqrt(2)/2, 0, -sqrt(2)), 0.2, M_PI/8));
		//objects.push_back(new Triangle(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0)));
		for (auto t : cubeMaker()) {
			objects.push_back(new Triangle(t));
		}
		for (auto t : tetrahedronMaker()) {
			objects.push_back(new Triangle(t));
		}


		//for (int i = 0; i < 150; i++) 
			//objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, material));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = vec3(0.2, 0.2, 0.2) * (1 - dot(hit.normal, ray.dir));
		for(auto light : lights) {
			Ray shadowray(hit.position, light->pos - hit.position);
			Hit shadowhit = firstIntersect(shadowray);
			if (shadowhit.t < 0 || shadowhit.t > length(light->pos - hit.position)) {
				outRadiance = outRadiance + light->color;
			} 
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocationdisc

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image) 
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
