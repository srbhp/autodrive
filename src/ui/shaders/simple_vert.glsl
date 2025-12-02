#version 330
in vec3 in_position;
uniform mat4 m_proj, m_cam, m_model;
void main() {
    gl_Position = m_proj * m_cam * m_model * vec4(in_position, 1.0);
}
