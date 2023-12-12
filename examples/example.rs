use std::{iter, time::Instant};

use bytemuck::cast_slice;
use gamebox::classes::{item::Indices, Item};
use glam::{Mat4, Vec3};
use pollster::FutureExt;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    vertex_attr_array, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBinding, BufferBindingType,
    BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, CommandEncoderDescriptor,
    DeviceDescriptor, Features, FragmentState, IndexFormat, Instance, InstanceDescriptor, Limits,
    MultisampleState, Operations, PipelineLayoutDescriptor, PowerPreference, PrimitiveState,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    TextureViewDescriptor, VertexBufferLayout, VertexState, VertexStepMode,
};
use winit::{
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, KeyCode, PhysicalKey},
    window::WindowBuilder,
};

fn main() {
    env_logger::init();

    let item: Item =
        gamebox::read_file("C:/Users/Justin/Projects/gamebox/tests/big_palm_tree_low.Item.Gbx")
            .unwrap();

    let event_loop = EventLoop::new().unwrap();

    let window = WindowBuilder::new()
        .with_title("example")
        .build(&event_loop)
        .unwrap();

    let instance = Instance::new(InstanceDescriptor::default());

    let surface = unsafe { instance.create_surface(&window) }.unwrap();

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::None,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .block_on()
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                features: Features::default(),
                limits: Limits::downlevel_webgl2_defaults(),
            },
            None,
        )
        .block_on()
        .unwrap();

    let window_size = window.inner_size();

    let surface_configuration = surface
        .get_default_config(&adapter, window_size.width, window_size.height)
        .unwrap();

    surface.configure(&device, &surface_configuration);

    let view_proj_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&view_proj_bind_group_layout],
        push_constant_ranges: &[],
    });

    let shader_module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&render_pipeline_layout),
        vertex: VertexState {
            module: &shader_module,
            entry_point: "vs_main",
            buffers: &[VertexBufferLayout {
                array_stride: 12,
                step_mode: VertexStepMode::Vertex,
                attributes: &vertex_attr_array![0 => Float32x3],
            }],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &shader_module,
            entry_point: "fs_main",
            targets: &[Some(ColorTargetState {
                format: surface_configuration.format,
                blend: None,
                write_mask: ColorWrites::default(),
            })],
        }),
        multiview: None,
    });

    let view_proj_buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 64,
        usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let view_proj_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &view_proj_bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &view_proj_buffer,
                offset: 0,
                size: None,
            }),
        }],
    });

    let mut vertex_buffers = vec![];
    let mut index_buffers = vec![];
    let mut index_formats = vec![];
    let mut num_indices = vec![];

    for layer in item.layers() {
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(layer.positions()),
            usage: BufferUsages::VERTEX,
        });

        vertex_buffers.push(vertex_buffer);

        match layer.indices() {
            Indices::U16(indices) => {
                let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: cast_slice(indices),
                    usage: BufferUsages::INDEX,
                });

                index_buffers.push(index_buffer);
            }
        }

        match layer.indices() {
            Indices::U16(_) => index_formats.push(IndexFormat::Uint16),
        }

        match layer.indices() {
            Indices::U16(indices) => num_indices.push(indices.len() as u32),
        }
    }

    let mut proj = Mat4::IDENTITY;

    let mut minimized = false;

    let mut last_time = Instant::now();

    let mut camera = Camera::new();

    event_loop.set_control_flow(ControlFlow::Poll);

    event_loop
        .run(|event, elwt| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(new_size) => {
                    if new_size.width > 0 && new_size.height > 0 {
                        let surface_configuration = surface
                            .get_default_config(&adapter, new_size.width, new_size.height)
                            .unwrap();

                        surface.configure(&device, &surface_configuration);

                        proj = Mat4::perspective_rh(
                            45.0,
                            new_size.width as f32 / new_size.height as f32,
                            0.1,
                            100.0,
                        );

                        minimized = false;
                    } else {
                        minimized = true;
                    }
                }
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::KeyboardInput { event, .. } => {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        camera.process_keyboard(code, event.state);
                    }
                }
                _ => {}
            },
            Event::AboutToWait => {
                if !minimized {
                    let current_time = Instant::now();
                    let elapsed_time = current_time - last_time;
                    last_time = current_time;
                    let dt = elapsed_time.as_secs_f32();

                    let surface_texture = surface.get_current_texture().unwrap();

                    let texture_view = surface_texture
                        .texture
                        .create_view(&TextureViewDescriptor::default());

                    camera.update(dt);
                    let view = camera.matrix();

                    let view_proj = proj * view;

                    queue.write_buffer(
                        &view_proj_buffer,
                        0,
                        cast_slice(&view_proj.to_cols_array()),
                    );

                    let mut command_encoder =
                        device.create_command_encoder(&CommandEncoderDescriptor::default());

                    {
                        let mut render_pass =
                            command_encoder.begin_render_pass(&RenderPassDescriptor {
                                color_attachments: &[Some(RenderPassColorAttachment {
                                    view: &texture_view,
                                    resolve_target: None,
                                    ops: Operations::default(),
                                })],
                                ..Default::default()
                            });

                        render_pass.set_pipeline(&render_pipeline);

                        render_pass.set_bind_group(0, &view_proj_bind_group, &[]);

                        for i in 0..vertex_buffers.len() {
                            render_pass.set_vertex_buffer(0, vertex_buffers[i].slice(..));
                            render_pass
                                .set_index_buffer(index_buffers[i].slice(..), index_formats[i]);
                            render_pass.draw_indexed(0..num_indices[i], 0, 0..1);
                        }
                    }

                    let command_buffer = command_encoder.finish();

                    queue.submit(iter::once(command_buffer));

                    surface_texture.present();
                }
            }
            _ => {}
        })
        .unwrap();
}

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    speed: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_left: f32,
    amount_right: f32,
    amount_up: f32,
    amount_down: f32,
}

impl Camera {
    fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            speed: 20.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_left: 0.0,
            amount_right: 0.0,
            amount_down: 0.0,
            amount_up: 0.0,
        }
    }

    fn process_keyboard(&mut self, key_code: KeyCode, state: ElementState) {
        let amount = if state.is_pressed() { 1.0 } else { 0.0 };

        match key_code {
            KeyCode::KeyW => self.amount_forward = amount,
            KeyCode::KeyS => self.amount_backward = amount,
            KeyCode::KeyA => self.amount_left = amount,
            KeyCode::KeyD => self.amount_right = amount,
            KeyCode::Space => self.amount_up = amount,
            KeyCode::ShiftLeft => self.amount_down = amount,
            _ => {}
        }
    }

    fn update(&mut self, delta_time: f32) {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let forward = Vec3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vec3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        self.position +=
            forward * (self.amount_forward - self.amount_backward) * self.speed * delta_time;
        self.position += right * (self.amount_right - self.amount_left) * self.speed * delta_time;

        self.position.y += (self.amount_up - self.amount_down) * self.speed * delta_time;
    }

    fn matrix(&self) -> Mat4 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();

        Mat4::look_to_rh(
            self.position,
            Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize(),
            Vec3::Y,
        )
    }
}
