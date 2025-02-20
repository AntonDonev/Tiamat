using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tiamat.Models;

namespace Tiamat.Core.Services.Interfaces
{
    public interface IPositionService
    {
        public void CreatePosition(string Symbol, string Type, Account account, decimal Size, decimal Risk, DateTime OpenedAt, string Id);
        public void ClosePosition(string Id, decimal profit, decimal currentCapital, DateTime ClosedAt);

    }
}
