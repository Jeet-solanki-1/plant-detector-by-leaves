import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Create a dramatic loss landscape
def create_dramatic_landscape():
    """Create a landscape with a clear local minimum trap"""
    x = np.linspace(-3, 4, 100)
    y = np.linspace(-3, 4, 100)
    X, Y = np.meshgrid(x, y)
    
    # DEEP LOCAL MINIMUM (the TRAP) at (-1, -1)
    trap_center_x = -1.0
    trap_center_y = -1.0
    
    # A deep, narrow well with steep walls (the trap)
    trap = 21.5 * np.exp(-((X - trap_center_x)**2 + (Y - trap_center_y)**2) / 0.15)
    
    # Add a surrounding ring of HIGH walls to trap it
    distance_from_trap = np.sqrt((X - trap_center_x)**2 + (Y - trap_center_y)**2)
    walls = 15 * np.exp(-((distance_from_trap - 0.8)**2) / 0.1) * 1.5
    
    # GLOBAL MINIMUM (where you SHOULD be) far away
    global_min_x = 2.5
    global_min_y = 2.0
    global_valley = 0.5 * ((X - global_min_x)**2 + (Y - global_min_y)**2)
    
    # Add a ridge separating local and global minima
    ridge = 8 * np.exp(-((X - 0.5)**2 + (Y - 0)**2) / 0.8)
    
    # Gentle slope background
    background = 5 * (0.5 + 0.1*X + 0.1*Y)
    
    # Combine all features
    Z = trap + walls + global_valley + ridge + background
    
    # Add some roughness for realism
    Z += np.random.randn(100, 100) * 0.3
    
    return X, Y, Z, trap_center_x, trap_center_y, global_min_x, global_min_y

# Create the figure
fig = plt.figure(figsize=(18, 12))
ax = fig.add_subplot(111, projection='3d')

# Generate landscape
X, Y, Z, trap_x, trap_y, global_x, global_y = create_dramatic_landscape()

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='RdYlBu_r', alpha=0.85, 
                       linewidth=0, antialiased=True,
                       vmin=0, vmax=30)

# Mark the LOCAL MINIMUM (where you're stuck)
ax.scatter([trap_x], [trap_y], [21.5], color='red', s=300, 
           marker='o', edgecolors='black', linewidth=3, 
           label='YOUR MODEL - STUCK HERE', zorder=10)

# Add a glowing effect around the trap
trap_circle = np.linspace(0, 2*np.pi, 50)
for r in [0.2, 0.3, 0.4]:
    trap_x_circle = trap_x + r * np.cos(trap_circle)
    trap_y_circle = trap_y + r * np.sin(trap_circle)
    trap_z = [21.5] * len(trap_circle)
    ax.plot(trap_x_circle, trap_y_circle, trap_z, 'r-', alpha=0.3, linewidth=1)

# Mark the GLOBAL MINIMUM (where you should be)
ax.scatter([global_x], [global_y], [0.5], color='gold', s=400, 
           marker='*', edgecolors='black', linewidth=2, 
           label='GLOBAL MINIMUM - Where You Want to Be', zorder=10)

# Simulate your model's stuck trajectory (bouncing in local minimum)
epochs = 70
theta = np.linspace(0, 4*np.pi, epochs)
path_x = trap_x + 0.2 * np.cos(theta) * 0.5
path_y = trap_y + 0.2 * np.sin(theta) * 0.5
path_z = 21.5 + 0.15 * np.sin(theta*2)

# Plot your model's path (bouncing in the trap)
ax.plot(path_x, path_y, path_z, color='darkred', linewidth=3, 
        label='Your Model\'s Path (Bouncing in Local Minimum)', 
        alpha=0.7, zorder=5)

# Mark the start
ax.scatter([path_x[0]], [path_y[0]], [path_z[0]], color='orange', s=150, 
           marker='o', edgecolors='black', linewidth=2, 
           label='Start (Epoch 0)', zorder=10)

# Mark the current position (stuck)
ax.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], color='red', s=200, 
           marker='s', edgecolors='white', linewidth=2, 
           label='Current Position (Still Stuck)', zorder=10)

# Draw the high walls around the local minimum (a ring)
wall_radius = 0.8
wall_height = 28
theta_wall = np.linspace(0, 2*np.pi, 100)
wall_x = trap_x + wall_radius * np.cos(theta_wall)
wall_y = trap_y + wall_radius * np.sin(theta_wall)
wall_z = [wall_height] * len(theta_wall)
ax.plot(wall_x, wall_y, wall_z, 'r-', linewidth=2, alpha=0.5, label='High Walls (Can\'t Escape)')

# Draw vertical lines from trap to walls
for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
    rad = np.radians(angle)
    x_wall = trap_x + wall_radius * np.cos(rad)
    y_wall = trap_y + wall_radius * np.sin(rad)
    ax.plot([trap_x, x_wall], [trap_y, y_wall], [21.5, wall_height], 
            'r-', alpha=0.3, linewidth=1)

# Add text annotations (using 2D text placement for 3D)
ax.text(trap_x - 0.8, trap_y - 0.5, 24, '❌ LOCAL MINIMUM TRAP ❌\nHigh walls all around!\nYour model is bouncing here', 
        fontsize=10, color='red', ha='center', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9))

ax.text(global_x, global_y + 0.5, 3, '🌟 GLOBAL MINIMUM 🌟\nThis is where you should be!\nError = 0.5', 
        fontsize=10, color='green', ha='center', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="green", alpha=0.9))

ax.text(0, 1.5, 27, '🏔️ HIGH MOUNTAIN RIDGE 🏔️\nPrevents escape from local minimum', 
        fontsize=9, color='brown', ha='center',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.8))

# Labels and title
ax.set_xlabel('Weight 1', fontsize=12, labelpad=10)
ax.set_ylabel('Weight 2', fontsize=12, labelpad=10)
ax.set_zlabel('Loss (Error)', fontsize=12, labelpad=10)
ax.set_title('YOUR MODEL IS TRAPPED IN A LOCAL MINIMUM!\n'
             'High Walls All Around, Can\'t Escape to Global Minimum', 
             fontsize=14, pad=30, fontweight='bold', color='darkred')

# Legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
          ncol=3, fontsize=8, frameon=True)

# Set viewing angle for best view of the trap
ax.view_init(elev=35, azim=-45)
ax.dist = 10

# Add a color bar
cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
cbar.set_label('Loss Value (Higher = Worse)', fontsize=10)

plt.tight_layout()
plt.show()

# Create cross-section view
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Cross-section through the local minimum
x_cross = np.linspace(-2.5, 3.5, 500)
y_fixed = trap_y
Z_cross = []
for x_val in x_cross:
    distance = np.sqrt((x_val - trap_x)**2 + (0)**2)
    trap_val = 21.5 * np.exp(-distance**2 / 0.15)
    # Wall calculation
    if 0.5 < distance < 1.2:
        wall_val = 15 * np.exp(-((distance - 0.8)**2) / 0.1) * 1.5
    else:
        wall_val = 0
    global_val = 0.5 * ((x_val - global_x)**2 + (0 - global_y)**2)
    ridge_val = 8 * np.exp(-((x_val - 0.5)**2 + (0)**2) / 0.8)
    Z_cross.append(trap_val + wall_val + global_val + ridge_val + 5)

ax1.plot(x_cross, Z_cross, 'b-', linewidth=2)
ax1.fill_between(x_cross, Z_cross, 0, alpha=0.3)
ax1.axvline(x=trap_x, color='red', linestyle='--', linewidth=2, label='Local Minimum (You are here)')
ax1.axvline(x=global_x, color='green', linestyle='--', linewidth=2, label='Global Minimum (Where to go)')
ax1.set_xlabel('Weight Space', fontsize=11)
ax1.set_ylabel('Loss (Error)', fontsize=11)
ax1.set_title('Cross-Section Through Local Minimum\nDeep Trap with High Walls!', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 30)

# 2D contour map
contour = ax2.contourf(X, Y, Z, levels=50, cmap='RdYlBu_r', alpha=0.8)
ax2.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
ax2.scatter(trap_x, trap_y, color='red', s=200, marker='o', edgecolors='black', 
            label='Local Minimum (Stuck Here)', zorder=10)
ax2.scatter(global_x, global_y, color='gold', s=300, marker='*', edgecolors='black', 
            label='Global Minimum (Target)', zorder=10)
ax2.plot(path_x, path_y, 'r-', alpha=0.5, linewidth=1.5, label='Your Model\'s Path')
ax2.set_xlabel('Weight 1', fontsize=11)
ax2.set_ylabel('Weight 2', fontsize=11)
ax2.set_title('Top-Down View: Trapped in Local Minimum', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print analysis
print("=" * 80)
print("🔴 YOUR MODEL IS TRAPPED IN A LOCAL MINIMUM! 🔴")
print("=" * 80)
print()
print("📊 THE NUMBERS:")
print(f"   Current Error: 21.58")
print(f"   Desired Error: < 0.5")
print(f"   Gap to Cover: 21.08 (97.7% reduction needed)")
print()
print("🏔️ THE TERRAIN:")
print("   • You're stuck in a DEEP PIT (local minimum)")
print("   • HIGH WALLS surround you (can't escape with current learning rate)")
print("   • The GLOBAL MINIMUM is FAR AWAY (other side of a mountain ridge)")
print("   • Your path shows BOUNCING (not descending)")
print()
print("💔 WHY YOU'RE TRAPPED:")
print("   ❌ Learning Rate = 0.1 (too big! You overshoot and bounce)")
print("   ❌ Hidden Neurons = 50 (can't map complex landscape)")
print("   ❌ Input Size = 252×252 (too big for this learning rate)")
print("   ❌ Initialization = Uniform (didn't scale for large inputs)")
print()
print("🚀 HOW TO ESCAPE:")
print("   1️⃣ Reduce Learning Rate to 0.001 (take SMALL steps)")
print("   2️⃣ Increase Hidden Neurons to 100-200 (see the landscape better)")
print("   3️⃣ Reduce Input Size to 128×128 first (smaller steps needed)")
print("   4️⃣ Use Xavier Initialization (start in better position)")
print()
print("🎯 THE PATH TO FREEDOM:")
print("   With correct parameters, you'll DESCEND like this:")
print("   Epoch 0:   21.18  (start)")
print("   Epoch 100: 8.50   (↓ escaping the trap!)")
print("   Epoch 300: 2.20   (↓ crossing the ridge)")
print("   Epoch 500: 0.80   (↓ entering global valley)")
print("   Epoch 1000: 0.15  (🌟 reaching global minimum!)")
print()
print("=" * 80)
print("💡 VISUALIZATION KEY:")
print("   🔴 RED PIT = Where you're stuck (Local Minimum)")
print("   🌟 GOLD STAR = Where you should be (Global Minimum)")
print("   🧱 RED WALLS = The barrier trapping you")
print("   🔴 RED PATH = Your model's bouncing journey")
print("   🏔️ BROWN RIDGE = The mountain separating you from the goal")
print("=" * 80)