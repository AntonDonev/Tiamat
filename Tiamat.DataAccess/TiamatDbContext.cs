using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using Tiamat.Models;
using System;

namespace Tiamat.DataAccess
{
    public class TiamatDbContext : IdentityDbContext<User, IdentityRole<Guid>, Guid>
    {
        public TiamatDbContext(DbContextOptions<TiamatDbContext> options)
            : base(options)
        {
        }

        public DbSet<Account> Accounts { get; set; }
        public DbSet<AccountSetting> AccountSettings { get; set; }
        public DbSet<Position> Positions { get; set; }
        public DbSet<AccountPosition> AccountPositions { get; set; }

        // Add these DbSets
        public DbSet<Notification> Notifications { get; set; }
        public DbSet<NotificationUser> NotificationUsers { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // Existing setup
            modelBuilder.Entity<User>()
                .HasMany(u => u.AccountSettings)
                .WithOne(a => a.User)
                .HasForeignKey(a => a.UserId)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<User>()
                .HasMany(u => u.Accounts)
                .WithOne(a => a.User)
                .HasForeignKey(a => a.UserId)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<Account>()
                .HasOne(a => a.AccountSetting)
                .WithMany(s => s.Accounts)
                .HasForeignKey(a => a.AccountSettingsId)
                .OnDelete(DeleteBehavior.Restrict);

            modelBuilder.Entity<AccountPosition>()
                .HasOne(ap => ap.Account)
                .WithMany(a => a.AccountPositions)
                .HasForeignKey(ap => ap.AccountId)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<AccountPosition>()
                .HasOne(ap => ap.Position)
                .WithMany(p => p.AccountPositions)
                .HasForeignKey(ap => ap.PositionId)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<AccountPosition>()
                .HasOne(ap => ap.Account)
                .WithMany(a => a.AccountPositions)
                .HasForeignKey(ap => ap.AccountId)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<AccountPosition>()
                .HasOne(ap => ap.Position)
                .WithMany(p => p.AccountPositions)
                .HasForeignKey(ap => ap.PositionId)
                .OnDelete(DeleteBehavior.Restrict);

            modelBuilder.Entity<Account>()
                .Property(a => a.HighestCapital)
                .HasPrecision(18, 2);

            modelBuilder.Entity<Account>()
                .Property(a => a.InitialCapital)
                .HasPrecision(18, 2);

            modelBuilder.Entity<Account>()
                .Property(a => a.CurrentCapital)
                .HasPrecision(18, 2);

            modelBuilder.Entity<Account>()
                .Property(a => a.LowestCapital)
                .HasPrecision(18, 2);

            modelBuilder.Entity<AccountPosition>()
                .Property(ap => ap.Size)
                .HasPrecision(18, 2);

            modelBuilder.Entity<AccountPosition>()
                .Property(ap => ap.Risk)
                .HasPrecision(18, 2);

            modelBuilder.Entity<AccountPosition>()
                .Property(ap => ap.Result)
                .HasPrecision(18, 2);

            modelBuilder.Entity<NotificationUser>()
                .HasKey(nu => new { nu.NotificationId, nu.UserId });

            modelBuilder.Entity<NotificationUser>()
                .HasOne(nu => nu.Notification)
                .WithMany(n => n.NotificationUsers)
                .HasForeignKey(nu => nu.NotificationId)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<NotificationUser>()
                .HasOne(nu => nu.User)
                .WithMany(u => u.NotificationUsers)
                .HasForeignKey(nu => nu.UserId)
                .OnDelete(DeleteBehavior.Cascade);
        }
    }
}
