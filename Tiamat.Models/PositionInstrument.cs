using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tiamat.Models
{
    public class PositionInstrument
    {

        [ForeignKey(nameof(Position))]
        public Guid PositionId { get; set; }
        public Position Position { get; set; }

        [ForeignKey(nameof(Instrument))]
        public int InstrumentId { get; set; }
        public Instrument Instrument { get; set; }
        public PositionInstrument() { }
        public PositionInstrument(Position position, Instrument instrument)
        {
            PositionId = position.Id;
            Position = position;
            InstrumentId = instrument.Id;
            Instrument = instrument;
        }
    }
}
