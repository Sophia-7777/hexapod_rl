使用手册：
1 查看训练过程中每个步骤的奖励和其他指标的变化曲线：
    (bishe) youcan7777@youcan7777-ThinkBook-14-G5-IRH:~/hexapod_rl_2/hexapod_tensorboard$ 
    tensorboard --logdir=./
    浏览器中输入http://localhost:6006/
    SCALARS 选项卡显示各种变化曲线
    
 2 直接通过mujoco运行xml
 	cd
 	python -m mujoco.viewer --mjcf=./A.xml

3 播放录取视频
    python mujoco_record.py

    将png转换为视频
    ffmpeg -y \
    -framerate 50 \
    -i frames/%03d.png \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -profile:v baseline \
    -level 3.0 \
    ok.mp4

    播放视频
    mpv ok.mp4

4 播放walk视频
    python walk_record.py
    mpv hexapod_walk.mp4

代码结构说明：
1 ppo代码流
    程序启动
        ↓
    make_env()
        ↓
    HexapodEnv.__init__()
        ↓
    HexapodEnv.seed()
        ↓
    model.learn()
        ↓
    env.reset()
        ↓
    for step in rollout:
            action = policy(obs)
            env.step(action)
            if done:
                env.reset()

2 ppo数据流
    state_t
    ↓
    policy
    ↓
    action_t
    ↓
    env.step(action)
    ↓
    next_state
    reward
    done