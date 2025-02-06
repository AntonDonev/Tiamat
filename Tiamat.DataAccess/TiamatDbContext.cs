using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tiamat.Models;

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
        public DbSet<Instrument> Instruments { get; set; }
        public DbSet<Position> Positions { get; set; }
        public DbSet<PositionInstrument> PositionInstruments { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

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

            modelBuilder.Entity<PositionInstrument>()
                .HasKey(pi => new { pi.PositionId, pi.InstrumentId });

            modelBuilder.Entity<PositionInstrument>()
                .HasOne(pi => pi.Position)
                .WithMany(p => p.PositionInstruments)
                .HasForeignKey(pi => pi.PositionId);

            modelBuilder.Entity<PositionInstrument>()
                .HasOne(pi => pi.Instrument)
                .WithMany(i => i.PositionInstruments)
                .HasForeignKey(pi => pi.InstrumentId);

            modelBuilder.Entity<Position>()
                .HasOne(p => p.Account)
                .WithMany(a => a.Positions)
                .HasForeignKey(p => p.AccountId)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<Account>()
                .Property(a => a.HighestCapital)
                .HasPrecision(18, 2);

            modelBuilder.Entity<Account>()
                .Property(a => a.InitialCapital)
                .HasPrecision(18, 2);

            modelBuilder.Entity<Account>()
                .Property(a => a.LowestCapital)
                .HasPrecision(18, 2);

            modelBuilder.Entity<Position>()
                .Property(p => p.Result)
                .HasPrecision(18, 2);

            modelBuilder.Entity<Position>()
                .Property(p => p.Risk)
                .HasPrecision(18, 2);

            modelBuilder.Entity<Position>()
                .Property(p => p.Size)
                .HasPrecision(18, 2);
        }
    }
}
