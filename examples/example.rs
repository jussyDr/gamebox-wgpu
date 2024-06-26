use std::{
    fs::File,
    iter,
    mem::size_of,
    num::NonZeroU64,
    sync::Arc,
    time::{Duration, Instant},
};

use bytemuck::{cast_slice, NoUninit};
use ddsfile::{Dds, DxgiFormat};
use futures::executor::block_on;
use gamebox::{
    engines::{
        game_data::item::ItemModel,
        plug::{material::Material, visual_indexed_triangles::Indices},
    },
    read::HeaderOptions,
    Item,
};
use glam::{Mat4, Vec3};
use wgpu::{
    include_wgsl,
    util::{BufferInitDescriptor, DeviceExt, TextureDataOrder},
    vertex_attr_array, Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBinding, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState,
    ColorWrites, CommandEncoderDescriptor, Device, DeviceDescriptor, Extent3d, Features,
    FragmentState, IndexFormat, Instance, InstanceDescriptor, Limits, MultisampleState, Operations,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    RequestAdapterOptions, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, Surface,
    Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureView, TextureViewDescriptor, TextureViewDimension, VertexBufferLayout, VertexState,
    VertexStepMode,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

fn main() {
    env_logger::init();

    let mut app = App { state: None };

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app).unwrap();
}

struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let instance = Instance::new(InstanceDescriptor::default());

        let surface = instance.create_surface(Arc::clone(&window)).unwrap();

        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .unwrap();

        let (device, queue) = block_on(adapter.request_device(
            &DeviceDescriptor {
                label: None,
                required_features: Features::TEXTURE_COMPRESSION_BC,
                required_limits: Limits::default(),
            },
            None,
        ))
        .unwrap();

        let window_size = window.inner_size();

        let surface_config = surface
            .get_default_config(&adapter, window_size.width, window_size.height)
            .unwrap();

        surface.configure(&device, &surface_config);

        let camera_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(16 * 4),
                    },
                    count: None,
                }],
            });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&camera_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader_module = device.create_shader_module(include_wgsl!("example.wgsl"));

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: "vert_main",
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x2],
                }],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: "frag_main",
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format: surface_config.format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 16 * 4,
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &camera_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &camera_buffer,
                    offset: 0,
                    size: NonZeroU64::new(16 * 4),
                }),
            }],
        });

        let camera = (Camera::new(), camera_buffer, camera_bind_group);

        let item: Item = gamebox::read_file("big_palm_tree_low.Item.Gbx").unwrap();

        let mut buffers = vec![];

        match item.model() {
            ItemModel::Entity(model) => {
                let model = model.static_object_model().solid_to_model();

                for mesh in model.meshes() {
                    let positions = mesh.vertex_stream().positions();
                    let texcoords = mesh.vertex_stream().texcoords();

                    let mut vertices = vec![];

                    for i in 0..positions.len() {
                        let position = &positions[i];
                        let texcoord = &texcoords[i];

                        vertices.push(Vertex {
                            pos: [position.x, position.y, position.z],
                            texcoord: [texcoord.x, texcoord.y],
                        });
                    }

                    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
                        label: None,
                        contents: cast_slice(&vertices),
                        usage: BufferUsages::VERTEX,
                    });

                    let indices = mesh.index_buffer().indices();

                    match indices {
                        Indices::U16(indices) => {
                            let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
                                label: None,
                                contents: cast_slice(indices),
                                usage: BufferUsages::INDEX,
                            });

                            buffers.push((vertex_buffer, index_buffer, indices.len() as u32));
                        }
                    }
                }
            }
            _ => panic!(),
        }

        let material: Material = gamebox::Reader::new()
            .read_header(HeaderOptions::Skip {
                assume_size_zero: true,
            })
            .read_file("GameData/Stadium/Media/Modifier/PlatformGrass/PlatformTech.Material.Gbx")
            .unwrap();

        let texture_path = material.custom().unwrap().diffuse_texture_path();

        let texture: gamebox::engines::plug::texture::Texture = gamebox::Reader::new()
            .read_header(HeaderOptions::Skip {
                assume_size_zero: true,
            })
            .read_file(format!("GameData/{}", texture_path.display()))
            .unwrap();

        let image_path = texture.image_path();

        let image = Dds::read(
            File::open(format!(
                "GameData/Stadium/Media/Texture/{}",
                image_path.display()
            ))
            .unwrap(),
        )
        .unwrap();

        let texture_format = match image.get_dxgi_format().unwrap() {
            DxgiFormat::BC1_UNorm_sRGB => TextureFormat::Bc1RgbaUnormSrgb,
            _ => panic!(),
        };

        let texture = device.create_texture_with_data(
            &queue,
            &TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: image.get_width(),
                    height: image.get_height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: texture_format,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            TextureDataOrder::default(),
            &image.get_data(0).unwrap(),
        );

        let texture_view = texture.create_view(&TextureViewDescriptor::default());

        let sampler = device.create_sampler(&SamplerDescriptor::default());

        let texture_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &texture_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        let texture = (texture, texture_view, sampler, texture_bind_group);

        self.state = Some(State {
            window,
            renderer: Renderer {
                surface,
                adapter,
                device,
                queue,
                render_pipeline,
                camera,
                buffers,
                texture,
            },
            last_render_time: Instant::now(),
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = self.state.as_mut().expect("state is None");

        match event {
            WindowEvent::Resized(size) => {
                let surface_config = state
                    .renderer
                    .surface
                    .get_default_config(&state.renderer.adapter, size.width, size.height)
                    .unwrap();

                state
                    .renderer
                    .surface
                    .configure(&state.renderer.device, &surface_config);
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    state.renderer.camera.0.process_keyboard(code, event.state);
                }
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let state = self.state.as_mut().expect("state is None");

        if let DeviceEvent::MouseMotion { delta } = event {
            state.renderer.camera.0.process_mouse(delta.0, delta.1);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let state = self.state.as_mut().expect("state is None");

        let now = Instant::now();
        let dt = now.duration_since(state.last_render_time);
        state.last_render_time = now;

        state.renderer.camera.0.update_camera(dt);

        let window_size = state.window.inner_size();

        state.renderer.queue.write_buffer(
            &state.renderer.camera.1,
            0,
            cast_slice(
                &state
                    .renderer
                    .camera
                    .0
                    .matrix(window_size.width as f32 / window_size.height as f32)
                    .to_cols_array(),
            ),
        );

        let surface_texture = state.renderer.surface.get_current_texture().unwrap();

        let surface_texture_view = surface_texture
            .texture
            .create_view(&TextureViewDescriptor::default());

        let mut command_encoder = state
            .renderer
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        {
            let mut render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &surface_texture_view,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&state.renderer.render_pipeline);

            render_pass.set_bind_group(0, &state.renderer.camera.2, &[]);
            render_pass.set_bind_group(1, &state.renderer.texture.3, &[]);

            for (vertex_buffer, index_buffer, num_indices) in &state.renderer.buffers {
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint16);
                render_pass.draw_indexed(0..*num_indices, 0, 0..1);
            }
        }

        let command_buffer = command_encoder.finish();

        state.renderer.queue.submit(iter::once(command_buffer));

        surface_texture.present();
    }
}

struct State {
    window: Arc<Window>,
    renderer: Renderer,
    last_render_time: Instant,
}

struct Renderer {
    surface: Surface<'static>,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    render_pipeline: RenderPipeline,
    camera: (Camera, Buffer, BindGroup),
    buffers: Vec<(Buffer, Buffer, u32)>,
    texture: (Texture, TextureView, Sampler, BindGroup),
}

#[repr(C)]
#[derive(Clone, Copy, NoUninit)]
struct Vertex {
    pos: [f32; 3],
    texcoord: [f32; 2],
}

struct Camera {
    pos: Vec3,
    yaw: f32,
    pitch: f32,

    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}

const SAFE_FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2 - 0.0001;

impl Camera {
    fn new() -> Self {
        Self {
            pos: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,

            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed: 20.0,
            sensitivity: 2.0,
        }
    }

    fn matrix(&self, aspect_ratio: f32) -> Mat4 {
        let projection =
            Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, aspect_ratio, 0.1, 1000.0);

        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();

        projection
            * Mat4::look_to_rh(
                self.pos,
                Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize(),
                Vec3::Y,
            )
    }

    fn process_keyboard(&mut self, key: KeyCode, state: ElementState) -> bool {
        let amount = if state == ElementState::Pressed {
            1.0
        } else {
            0.0
        };

        match key {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.amount_forward = amount;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.amount_backward = amount;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.amount_left = amount;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.amount_right = amount;
                true
            }
            KeyCode::Space => {
                self.amount_up = amount;
                true
            }
            KeyCode::ShiftLeft => {
                self.amount_down = amount;
                true
            }
            _ => false,
        }
    }

    fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }

    fn update_camera(&mut self, dt: Duration) {
        let dt = dt.as_secs_f32();

        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let forward = Vec3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vec3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        self.pos += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        self.pos += right * (self.amount_right - self.amount_left) * self.speed * dt;

        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();
        let scrollward = Vec3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        self.pos += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        self.pos.y += (self.amount_up - self.amount_down) * self.speed * dt;

        self.yaw += self.rotate_horizontal * self.sensitivity * dt;
        self.pitch += -self.rotate_vertical * self.sensitivity * dt;

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        self.pitch = self.pitch.clamp(-SAFE_FRAC_PI_2, SAFE_FRAC_PI_2);
    }
}
